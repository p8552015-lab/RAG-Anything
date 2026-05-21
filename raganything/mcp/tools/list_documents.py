"""rag_list_documents — 列出已加入 RAG 知識庫的文件

直接讀 LightRAG ``doc_status`` storage，不自己 walk filesystem（DRY）。
LightRAG 1.4.16 用 ``get_docs_paginated(status_filter, page, page_size)``
取代舊的 ``get_all``；本 tool 把 ``offset/limit`` 翻譯成 LightRAG 的 page 概念。

過濾 LightRAG 標記為 duplicate 的記錄（``dup-`` 前綴）以免混淆。
"""

from __future__ import annotations

from typing import Literal, Optional

from lightrag.base import DocStatus
from mcp.server.fastmcp import FastMCP

from raganything.mcp import rag_instance
from raganything.mcp.schemas import DocumentEntry, ListDocumentsOutput


StatusFilter = Literal[
    "all", "processed", "processing", "pending", "preprocessed", "failed"
]


def _map_status(value: str) -> Optional[DocStatus]:
    """字串 → DocStatus enum；'all' → None（表示不過濾）。"""
    v = value.lower()
    if v == "all":
        return None
    for ds in DocStatus:
        if ds.value == v:
            return ds
    raise ValueError(
        f"Unknown status_filter: {value!r}. "
        f"Allowed: all, {', '.join(ds.value for ds in DocStatus)}"
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_list_documents(
        status_filter: StatusFilter = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> ListDocumentsOutput:
        """列出已加入知識庫的文件。

        Args:
            status_filter: ``all`` / ``processed`` / ``processing`` /
                ``pending`` / ``preprocessed`` / ``failed``。
            limit: 回傳筆數上限（同 LightRAG page_size）。
            offset: 分頁起點（會被換算為 page = offset // limit + 1）。

        Returns:
            ListDocumentsOutput: ``total`` 為套用 filter 後的總數，
            ``documents`` 為當頁項目（已過濾 LightRAG 的 ``dup-*`` 記錄）。
        """
        rag = await rag_instance.get_rag()
        status_enum = _map_status(status_filter)

        page_size = max(limit, 1)
        page = (offset // page_size) + 1

        items, total = await rag.lightrag.doc_status.get_docs_paginated(
            status_filter=status_enum,
            page=page,
            page_size=page_size,
            sort_field="updated_at",
            sort_direction="desc",
        )

        entries: list[DocumentEntry] = []
        for doc_id, info in items:
            if doc_id.startswith("dup-"):
                continue
            status_val = info.status.value if hasattr(info.status, "value") else str(info.status)
            entries.append(
                DocumentEntry(
                    doc_id=doc_id,
                    file_name=getattr(info, "file_path", "unknown") or "unknown",
                    status=status_val,
                    created_at=getattr(info, "created_at", "") or "",
                    updated_at=getattr(info, "updated_at", "") or "",
                    chunk_count=int(getattr(info, "chunks_count", 0) or 0),
                )
            )

        return ListDocumentsOutput(total=total, documents=entries)
