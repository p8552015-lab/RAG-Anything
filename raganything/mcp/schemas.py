"""raganything.mcp.schemas — MCP tool 出參 Pydantic 模型

只放 Output 模型；Input 由 tool function signature + type hints 推斷，
不重複定義避免 DRY 違反。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StatusOutput(BaseModel):
    """`rag_get_status` 出參。"""

    working_dir: str = Field(..., description="LightRAG 儲存路徑")
    parser: str = Field(..., description="文件解析器名稱 (mineru / docling / paddleocr)")
    llm_model: str = Field(..., description="LLM 模型 tag")
    embedding_model: str = Field(..., description="Embedding 模型 tag")
    embedding_dim: int = Field(..., description="Embedding 向量維度")
    vision_model: Optional[str] = Field(
        None, description="Vision 模型 tag (未設則為 None；VLM enhanced query 自動停用)"
    )
    document_count: int = Field(0, description="已索引文件數")
    initialized: bool = Field(False, description="RAGAnything 實例是否已 lazy init")
    raganything_version: str = Field(..., description="raganything 套件版本")


class AddDocumentOutput(BaseModel):
    """`rag_add_document` 出參。"""

    doc_id: str = Field(..., description="文件 ID (使用者指定或內容 hash 自動產生)")
    file_name: str = Field(..., description="檔名 (供查詢時 citation 用)")
    content_blocks: int = Field(..., description="parse 出來的 content block 總數")
    text_chars: int = Field(..., description="純文字內容字元數")
    multimodal_items: int = Field(..., description="圖片 / 表格 / 公式項數")
    chunk_count: int = Field(0, description="切出的 chunk 數")
    elapsed_seconds: float = Field(..., description="總處理時間 (秒)")
    parser_used: str = Field(..., description="實際使用的解析器名稱")


class QueryOutput(BaseModel):
    """`rag_query` 與 `rag_query_multimodal` 出參。"""

    answer: str = Field(..., description="LLM 回答 (含 citation tag)")
    mode: str = Field(..., description="實際使用的 query mode")
    elapsed_seconds: float = Field(..., description="查詢時間 (秒)")


class DocumentEntry(BaseModel):
    """`rag_list_documents` 中單筆文件資訊。"""

    doc_id: str
    file_name: str
    status: str = Field(..., description="processed / pending / processing / failed")
    created_at: str
    updated_at: str
    chunk_count: int = 0


class ListDocumentsOutput(BaseModel):
    """`rag_list_documents` 出參。"""

    total: int = Field(..., description="符合條件的文件總數 (套用 filter 後、分頁前)")
    documents: list[DocumentEntry] = Field(default_factory=list)
