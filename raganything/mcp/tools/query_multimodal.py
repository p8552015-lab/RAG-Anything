"""rag_query_multimodal — 多模態查詢 (帶圖 / 表 / 公式)

接 ``QueryMixin.aquery_with_multimodal``。multimodal_content 為空時自動退回
``aquery``（這是 RAG-Anything 內建行為，本 wrapper 直接透傳）。
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from raganything.mcp import rag_instance
from raganything.mcp.schemas import QueryOutput


QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class MultimodalItem(BaseModel):
    """單一多模態項目；type 決定其他欄位是否要填。"""

    type: Literal["image", "table", "equation"]
    img_path: Optional[str] = Field(None, description="image type 必填，絕對路徑")
    table_data: Optional[str] = Field(
        None, description="table type 用：markdown 或 csv 風格的表格內容"
    )
    table_caption: Optional[str] = None
    latex: Optional[str] = Field(None, description="equation type 必填，LaTeX 公式")
    equation_caption: Optional[str] = None


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_query_multimodal(
        query: str,
        multimodal_content: list[MultimodalItem],
        mode: QueryMode = "hybrid",
    ) -> QueryOutput:
        """以額外的多模態內容（圖片、表格、公式）作為查詢上下文。

        Args:
            query: 查詢文字。
            multimodal_content: 列表，每筆描述一個圖/表/公式。
            mode: 同 rag_query 的查詢模式。

        Returns:
            QueryOutput: LLM 結合知識庫 + 提供的多模態內容後的回答。
        """
        start = time.time()
        rag = await rag_instance.get_rag()

        items: list[dict[str, Any]] = [item.model_dump(exclude_none=True) for item in multimodal_content]
        answer = await rag.aquery_with_multimodal(query, multimodal_content=items, mode=mode)

        return QueryOutput(
            answer=answer or "",
            mode=mode,
            elapsed_seconds=round(time.time() - start, 2),
        )
