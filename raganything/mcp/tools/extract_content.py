"""rag_extract_content — 純抽取文件結構化內容，不入庫

對 ``rag.parse_document`` 的結果做過濾分類，給 Claude 直接看，不汙染知識庫。
適合「我只想看這份 PDF 的表格」這類需求。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from raganything.mcp import rag_instance


ContentType = Literal["text", "table", "image", "equation"]


class ExtractedItem(BaseModel):
    type: str
    page_idx: int | None = None
    content: dict[str, Any] = Field(default_factory=dict, description="依 type 不同")


class ExtractContentOutput(BaseModel):
    file_path: str
    parser_used: str
    items: list[ExtractedItem]
    counts: dict[str, int] = Field(default_factory=dict)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_extract_content(
        file_path: str,
        types: list[ContentType] | None = None,
        parse_method: str | None = None,
    ) -> ExtractContentOutput:
        """解析文件並回傳結構化內容，**不**入 RAG 知識庫。

        Args:
            file_path: 文件路徑。
            types: 只回這些類型；None 表示全部（text/table/image/equation）。
            parse_method: 'auto' / 'ocr' / 'txt'，未指定走 config 預設。

        Returns:
            ExtractContentOutput: items 為過濾後的 content_list，counts 為各類數量統計。
        """
        rag = await rag_instance.get_rag()

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")

        wanted = set(types) if types else {"text", "table", "image", "equation"}

        content_list, _ = await rag.parse_document(
            str(path),
            output_dir=rag.config.parser_output_dir,
            parse_method=parse_method or rag.config.parse_method,
            display_stats=False,
        )

        items: list[ExtractedItem] = []
        counts: dict[str, int] = {}
        for block in content_list:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "unknown"))
            counts[block_type] = counts.get(block_type, 0) + 1
            if block_type not in wanted:
                continue
            page_idx = block.get("page_idx")
            payload = {k: v for k, v in block.items() if k not in {"type", "page_idx"}}
            items.append(
                ExtractedItem(
                    type=block_type,
                    page_idx=int(page_idx) if isinstance(page_idx, int) else None,
                    content=payload,
                )
            )

        return ExtractContentOutput(
            file_path=str(path),
            parser_used=rag.config.parser,
            items=items,
            counts=counts,
        )
