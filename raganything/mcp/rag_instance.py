"""raganything.mcp.rag_instance — RAGAnything singleton + lazy async init

模組級單例，第一次任何 tool 呼叫才會：
1. 建構 LLM / Embedding / Vision 函式
2. 跑啟動自檢（Ollama 連通、模型存在、實測 embedding 維度）
3. 建構 RAGAnything 並 cache

關掉時透過 ``shutdown()`` 顯式呼叫 ``finalize_storages()``，避免 ``atexit`` 在
stdio MCP server 結束時踩到事件迴圈狀態的 race。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from raganything import RAGAnything, RAGAnythingConfig
from raganything.mcp.providers import (
    build_embedding_func,
    build_llm_func,
    build_vision_func,
)

logger = logging.getLogger(__name__)

# 模組級狀態：刻意不放進 class，避免狀態跨多個實例混淆。
_rag: Optional[RAGAnything] = None
_init_lock: Optional[asyncio.Lock] = None


def _get_init_lock() -> asyncio.Lock:
    """Lazy 建立 lock，避免 import 時就要有 event loop。"""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def get_current_rag() -> Optional[RAGAnything]:
    """同步取得當前 singleton（未初始化則 None），供 status tool 等使用。"""
    return _rag


async def _startup_check(embedding_func) -> None:
    """連通檢查 + 模型存在性 + embedding 維度核對。

    嚴格 fail-fast；任何項目失敗都 raise（CLAUDE.md 禁止降級）。
    """
    import ollama

    embed_host = os.environ["EMBEDDING_BINDING_HOST"]
    embed_model = os.environ["EMBEDDING_MODEL"]
    expected_dim = int(os.environ["EMBEDDING_DIM"])
    llm_model = os.environ["LLM_MODEL"]

    client = ollama.AsyncClient(host=embed_host)
    try:
        models_response = await client.list()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama 連線失敗 host={embed_host}: {exc}"
        ) from exc

    available = {m.model for m in models_response.models}
    for required in (embed_model, llm_model):
        prefix = required.split(":")[0]
        if not any(m.startswith(prefix) for m in available):
            raise RuntimeError(
                f"必要模型 {required} 不在 {embed_host} 上。"
                f"請在 Ollama 主機執行: ollama pull {required}"
            )

    test_vec = await embedding_func.func(["dim_probe"])
    actual_dim = int(test_vec.shape[-1]) if hasattr(test_vec, "shape") else (
        len(test_vec[0]) if test_vec else 0
    )
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"Embedding 維度不符：EMBEDDING_DIM={expected_dim} 但 {embed_model} "
            f"實際回 {actual_dim}。修正 .env 或重新確認模型。"
        )
    logger.info(
        "startup_check OK: ollama=%s, llm=%s, embed=%s, dim=%d",
        embed_host,
        llm_model,
        embed_model,
        actual_dim,
    )


async def get_rag() -> RAGAnything:
    """取得 RAGAnything singleton；首次呼叫做啟動自檢與建構。"""
    global _rag
    if _rag is not None:
        return _rag

    async with _get_init_lock():
        if _rag is not None:
            return _rag

        # 註冊自訂遠端 parser，讓 PARSER=mineru-remote 可用。
        # 無條件註冊（import 成本低），register_parser 對重複名稱會覆寫。
        try:
            from raganything.parser import register_parser
            from raganything.mcp.mineru_remote import MineruRemoteParser

            register_parser("mineru-remote", MineruRemoteParser)
        except Exception as exc:  # 不該因為註冊失敗讓整個 server 起不來
            logger.warning("register mineru-remote parser 失敗: %s", exc)

        llm_func = build_llm_func()
        embedding_func = build_embedding_func()
        vision_func = build_vision_func()

        await _startup_check(embedding_func)

        working_dir = os.environ.get("WORKING_DIR") or os.path.abspath("./rag_storage")
        # 重要 (R11)：RAGAnythingConfig 的 field default 在 module load 時 evaluate，
        # 比 _setup_env() 早，所以 env 值不會被自動套用。這裡顯式從 env 覆寫
        # parser / parse_method / parser_output_dir，避免使用者在 .env 改了
        # PARSER 卻沒生效的 silent bug。
        def _truthy(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=os.environ.get("PARSER") or "mineru",
            parse_method=os.environ.get("PARSE_METHOD") or "auto",
            parser_output_dir=os.environ.get("OUTPUT_DIR") or "./output",
            enable_image_processing=_truthy("ENABLE_IMAGE_PROCESSING", True),
            enable_table_processing=_truthy("ENABLE_TABLE_PROCESSING", True),
            enable_equation_processing=_truthy("ENABLE_EQUATION_PROCESSING", True),
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embedding_func,
        )
        await rag._ensure_lightrag_initialized()
        logger.info("RAGAnything singleton ready, working_dir=%s", working_dir)
        _rag = rag
        return rag


async def shutdown() -> None:
    """顯式關閉 storage，避免 atexit 在 stdio MCP 場景下踩到事件迴圈問題。"""
    global _rag
    if _rag is None:
        return
    try:
        await _rag.finalize_storages()
    finally:
        _rag = None
