"""raganything.mcp.providers — LLM / Embedding / Vision 函式工廠

所有設定從環境變數讀，缺必要變數時 fail-fast（遵守 CLAUDE.md「禁止降級」原則）。

關鍵 invariant：
* Embedding callable 一定回 ``numpy.ndarray``（R9：LightRAG query path 依賴 .size 屬性）
* 走 Ollama 原生 ``/api/embed``，host 不要帶 ``/v1``
* Vision 缺 ``VISION_MODEL`` 時回 ``None``，由呼叫端決定如何處理（禁止 silently fallback）
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc


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
        response = await client.embed(model=model, input=list(texts))
        return np.array(response.embeddings, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=max_token,
        func=embed_func,
    )


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
