"""rag_delete_document — 從 RAG 知識庫移除文件

走 LightRAG 的 ``adelete_by_doc_id``（會連帶清掉 chunks、entities、relations
中與該 doc 唯一相關的記錄）。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from raganything.mcp import rag_instance


class DeleteDocumentOutput(BaseModel):
    doc_id: str
    deleted: bool
    message: str = ""


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
            return DeleteDocumentOutput(
                doc_id=doc_id,
                deleted=False,
                message="找不到此 doc_id 於 doc_status",
            )

        await rag.lightrag.adelete_by_doc_id(doc_id)
        return DeleteDocumentOutput(
            doc_id=doc_id,
            deleted=True,
            message="LightRAG 已移除文件及其圖譜記錄",
        )
