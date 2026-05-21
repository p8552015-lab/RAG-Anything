"""rag_delete_document — 從 RAG 知識庫移除文件

走 LightRAG 的 ``adelete_by_doc_id``（會連帶清掉 chunks、entities、relations
中與該 doc 唯一相關的記錄）。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from raganything.mcp.cache import clear_query_response_cache
from raganything.mcp.cleanup import cleanup_document_residue
from raganything.mcp import rag_instance


class DeleteDocumentOutput(BaseModel):
    doc_id: str
    deleted: bool
    message: str = ""
    cleanup: dict[str, int] = Field(default_factory=dict)


def _cleanup_removed_anything(summary: dict[str, int]) -> bool:
    return any(value > 0 for key, value in summary.items() if key != "chunk_ids_found")


def _merge_cleanup_summaries(*summaries: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for summary in summaries:
        for key, value in summary.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_delete_document(doc_id: str) -> DeleteDocumentOutput:
        """從知識庫移除指定文件與其產生的 chunks / entities / relations。

        Args:
            doc_id: 文件 ID（由 rag_add_document 回傳或 rag_list_documents 取得）。

        Returns:
            DeleteDocumentOutput: deleted 表示 LightRAG 是否真的有移除記錄。
        """
        rag = await rag_instance.get_rag()

        existing = await rag.lightrag.doc_status.get_by_id(doc_id)
        if existing is None:
            cleanup = await cleanup_document_residue(rag, doc_id)
            if _cleanup_removed_anything(cleanup):
                cleanup["query_cache_deleted"] = await clear_query_response_cache(
                    rag.lightrag
                )
                return DeleteDocumentOutput(
                    doc_id=doc_id,
                    deleted=True,
                    message="doc_status 找不到，但已清除殘留的 chunks / graph / vector 記錄",
                    cleanup=cleanup,
                )
            return DeleteDocumentOutput(
                doc_id=doc_id,
                deleted=False,
                message="找不到此 doc_id 於 doc_status",
                cleanup=cleanup,
            )

        pre_cleanup = await cleanup_document_residue(
            rag,
            doc_id,
            delete_doc_records=False,
        )
        native_result = await rag.lightrag.adelete_by_doc_id(doc_id)
        post_cleanup = await cleanup_document_residue(rag, doc_id)
        cleanup = _merge_cleanup_summaries(pre_cleanup, post_cleanup)
        cleanup["query_cache_deleted"] = await clear_query_response_cache(rag.lightrag)
        native_status = getattr(native_result, "status", "unknown")
        native_message = getattr(native_result, "message", "")
        if native_status not in {"success", "not_found"}:
            return DeleteDocumentOutput(
                doc_id=doc_id,
                deleted=_cleanup_removed_anything(cleanup),
                message=(
                    f"LightRAG 刪除結果為 {native_status}: {native_message}; "
                    "已執行 MCP residual cleanup"
                ),
                cleanup=cleanup,
            )

        return DeleteDocumentOutput(
            doc_id=doc_id,
            deleted=True,
            message="LightRAG 已移除文件；MCP 已補做 residual cleanup",
            cleanup=cleanup,
        )
