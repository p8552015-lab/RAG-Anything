"""raganything.mcp.providers — LLM / Embedding / Vision 函式工廠

所有設定從環境變數讀，缺必要變數時 fail-fast（遵守 CLAUDE.md「禁止降級」原則）。

關鍵 invariant：
* Embedding callable 一定回 ``numpy.ndarray``（R9：LightRAG query path 依賴 .size 屬性）
* 走 Ollama 原生 ``/api/embed``，host 不要帶 ``/v1``
* Vision 缺 ``VISION_MODEL`` 時回 ``None``，由呼叫端決定如何處理（禁止 silently fallback）
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Callable, Optional, Sequence

import numpy as np
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """讀必要 env，缺值直接 raise。

    Raises:
        RuntimeError: 找不到變數或值為空字串。
    """
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"必要環境變數 {name} 未設定或為空。"
            "請檢查 .env 或 Claude Desktop config 的 env 區塊。"
        )
    return val


def build_llm_func() -> Callable[..., Any]:
    """建構 LLM 補完函式。

    使用 OpenAI 相容 chat completions endpoint，呼叫端要自行傳 prompt /
    system_prompt / history_messages 與其他 kwargs。
    """
    model = _require_env("LLM_MODEL")
    host = _require_env("LLM_BINDING_HOST")
    api_key = os.environ.get("LLM_BINDING_API_KEY") or "ollama"

    async def llm_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=host,
            api_key=api_key,
            **kwargs,
        )

    return llm_func


def build_embedding_func() -> EmbeddingFunc:
    """建構 Embedding 函式並包成 LightRAG ``EmbeddingFunc``。

    內部以 numpy.float32 ndarray 形式回傳（R9）。走 Ollama 原生 ``/api/embed``。
    """
    model = _require_env("EMBEDDING_MODEL")
    host = _require_env("EMBEDDING_BINDING_HOST")
    dim = int(_require_env("EMBEDDING_DIM"))
    max_token = int(os.environ.get("MAX_EMBED_TOKENS") or "8192")

    async def embed_func(texts: Sequence[str]) -> np.ndarray:
        import ollama

        client = ollama.AsyncClient(host=host)
        return await _embed_with_ollama_fallback(
            client=client,
            model=model,
            texts=texts,
            expected_dim=dim,
        )

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=max_token,
        func=embed_func,
    )


def _normalize_embedding_text(value: Any) -> str:
    """Normalize values before sending them to Ollama embedding.

    LightRAG storage should send strings, but failure recovery paths may pass
    non-string values. Avoid serializing Python/JSON NaN, because Ollama's Go
    JSON encoder rejects non-finite values with ``unsupported value: NaN``.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else str(value)
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            text = str(value)

    # Keep normal newlines/tabs, drop other control characters that can make
    # downstream JSON/debug output hard to reason about.
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    return text.strip()


def _short_embedding_snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _aggressive_embedding_text(text: str) -> str:
    """Drop punctuation that can trigger Ollama/bge-m3 NaN JSON failures."""
    stripped = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text, flags=re.UNICODE)
    stripped = " ".join(stripped.split())
    return stripped or text


def _first_embedding_token(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    return next((part.strip() for part in first_line.split() if part.strip()), "")


def _embedding_retry_candidates(text: str) -> list[str]:
    """Return progressively safer text variants for unstable Ollama embeddings.

    The remote bge-m3 endpoint can return a JSON NaN error for some otherwise
    valid Chinese entity-description strings. Short entity/key phrase variants
    still embed correctly, so final fallback preserves lookup usefulness instead
    of failing the whole document ingestion.
    """
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(text)
    add(_aggressive_embedding_text(text))

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    add(first_line)
    add(_aggressive_embedding_text(first_line))

    raw_first_token = _first_embedding_token(text)
    add(raw_first_token)
    add(_aggressive_embedding_text(raw_first_token))

    aggressive_first_token = _first_embedding_token(_aggressive_embedding_text(text))
    add(aggressive_first_token)

    return candidates


def _is_retryable_embedding_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "unsupported value: nan" in message
        or "status code: 500" in message
        or "internal server error" in message
        or "non-finite" in message
    )


def _as_finite_embedding_array(
    embeddings: Any,
    *,
    expected_count: int,
    expected_dim: int,
) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape != (expected_count, expected_dim):
        raise ValueError(
            "Embedding shape mismatch: "
            f"expected {(expected_count, expected_dim)} but got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Embedding response contained non-finite values")
    return array


async def _embed_once(
    client: Any,
    *,
    model: str,
    texts: Sequence[str],
    expected_dim: int,
) -> np.ndarray:
    response = await client.embed(model=model, input=list(texts))
    return _as_finite_embedding_array(
        response.embeddings,
        expected_count=len(texts),
        expected_dim=expected_dim,
    )


async def _embed_with_ollama_fallback(
    *,
    client: Any,
    model: str,
    texts: Sequence[Any],
    expected_dim: int,
) -> np.ndarray:
    """Embed texts with batch retry for Ollama NaN/500 failures.

    Ollama can fail an entire batch with ``unsupported value: NaN`` even when
    each item embeds successfully on its own. On those retryable failures, split
    the batch recursively. If a single item still fails, try progressively
    smaller text variants and then raise a contextual error instead of returning
    a bad vector.
    """
    normalized_texts = [_normalize_embedding_text(text) for text in texts]
    if not normalized_texts:
        return np.empty((0, expected_dim), dtype=np.float32)

    try:
        return await _embed_once(
            client,
            model=model,
            texts=normalized_texts,
            expected_dim=expected_dim,
        )
    except Exception as exc:
        if len(normalized_texts) == 1:
            text = normalized_texts[0]
            retry_candidates = _embedding_retry_candidates(text)

            if _is_retryable_embedding_error(exc):
                retry_errors: list[str] = [str(exc)]
                for retry_text in retry_candidates:
                    try:
                        return await _embed_once(
                            client,
                            model=model,
                            texts=[retry_text],
                            expected_dim=expected_dim,
                        )
                    except Exception as retry_exc:
                        retry_errors.append(str(retry_exc))
                raise RuntimeError(
                    "Ollama embed failed for one text after retry. "
                    f"model={model}, snippet={_short_embedding_snippet(text)!r}, "
                    f"errors={retry_errors}"
                ) from exc

            retry_text = _normalize_embedding_text(text)
            if retry_text != text:
                try:
                    return await _embed_once(
                        client,
                        model=model,
                        texts=[retry_text],
                        expected_dim=expected_dim,
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        "Ollama embed failed for one text after retry. "
                        f"model={model}, snippet={_short_embedding_snippet(retry_text)!r}, "
                        f"original_error={exc}, retry_error={retry_exc}"
                    ) from retry_exc
            raise

        if not _is_retryable_embedding_error(exc):
            raise

        logger.warning(
            "Ollama embed batch failed; retrying by splitting batch. "
            "model=%s, batch_size=%d, error=%s",
            model,
            len(normalized_texts),
            exc,
        )
        midpoint = len(normalized_texts) // 2
        left = await _embed_with_ollama_fallback(
            client=client,
            model=model,
            texts=normalized_texts[:midpoint],
            expected_dim=expected_dim,
        )
        right = await _embed_with_ollama_fallback(
            client=client,
            model=model,
            texts=normalized_texts[midpoint:],
            expected_dim=expected_dim,
        )
        return np.concatenate([left, right], axis=0)


def build_vision_func() -> Optional[Callable[..., Any]]:
    """建構 Vision 補完函式；未設 ``VISION_MODEL`` 時回 ``None``。

    支援三種輸入形式：
    * ``messages``: 完整 OpenAI 多模態 messages 結構（VLM enhanced query 使用）
    * ``image_data``: 單張 base64 圖片 + prompt
    * 純文字：fallback 給 LLM_MODEL 處理（同 raganything example 行為）
    """
    model = os.environ.get("VISION_MODEL")
    if not model:
        return None
    host = os.environ.get("VISION_BINDING_HOST") or _require_env("LLM_BINDING_HOST")
    api_key = os.environ.get("VISION_BINDING_API_KEY") or "ollama"
    text_fallback = build_llm_func()

    async def vision_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        image_data: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        if messages:
            return await openai_complete_if_cache(
                model=model,
                prompt="",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                base_url=host,
                api_key=api_key,
                **kwargs,
            )
        if image_data:
            built_messages: list[dict[str, Any]] = []
            if system_prompt:
                built_messages.append({"role": "system", "content": system_prompt})
            built_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            )
            return await openai_complete_if_cache(
                model=model,
                prompt="",
                system_prompt=None,
                history_messages=[],
                messages=built_messages,
                base_url=host,
                api_key=api_key,
                **kwargs,
            )
        return await text_fallback(
            prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs
        )

    return vision_func
