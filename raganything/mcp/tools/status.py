"""rag_get_status — 回報 server 配置與索引統計

故意不觸發 ``rag_instance.get_rag()`` 的 lazy init，避免每次 status 都跑啟動
自檢。只有 RAGAnything 已建構時才嘗試讀 ``doc_status``。
"""

from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP

from raganything import __version__ as raganything_version
from raganything.mcp import rag_instance
from raganything.mcp.schemas import StatusOutput

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_get_status() -> StatusOutput:
        """回報當前 MCP server 配置與索引統計。

        若 RAGAnything 尚未建構（沒任何 tool 被呼叫過），``initialized`` 為 False，
        ``document_count`` 維持 0。已建構時讀 ``lightrag.doc_status``，過濾掉
        LightRAG 自動標記的 duplicate 記錄。
        """
        rag = rag_instance.get_current_rag()
        initialized = rag is not None
        document_count = 0

        if initialized and rag is not None:
            try:
                # LightRAG 1.4.16: 用 get_status_counts 而非已不存在的 get_all
                counts = await rag.lightrag.doc_status.get_status_counts()
                # 排除 "failed" 是因為 LightRAG 對 dup-* duplicate 記錄會標 failed
                document_count = sum(
                    n for status, n in (counts or {}).items() if status != "failed"
                )
            except Exception as exc:
                logger.debug("doc_status.get_status_counts 失敗: %s", exc)

        return StatusOutput(
            working_dir=os.environ.get("WORKING_DIR", ""),
            parser=os.environ.get("PARSER") or "mineru",
            llm_model=os.environ.get("LLM_MODEL", ""),
            embedding_model=os.environ.get("EMBEDDING_MODEL", ""),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM") or "0"),
            vision_model=os.environ.get("VISION_MODEL") or None,
            document_count=document_count,
            initialized=initialized,
            raganything_version=raganything_version,
        )
