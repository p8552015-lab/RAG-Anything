"""rag_add_document — 把文件加入 RAG 知識庫 (B 方案)

不走 ``process_document_complete`` 以避開 R8 (RAGAnything 1.3.0 + LightRAG
1.4.16 的提前 upsert regression)。改自行 orchestrate：

    parse_document → separate_content → lightrag.ainsert → _process_multimodal_content

最終效果等價於 ``process_document_complete``，但 doc_status 是由 LightRAG ainsert
路徑建立，避免 ``filter_keys`` 把 doc_id 視為 duplicate。

支援副檔名：依 ``RAGAnythingConfig.supported_file_extensions`` 為準（PDF / Office /
TXT / MD / 圖片共 17 種）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from raganything.mcp import rag_instance
from raganything.mcp.schemas import AddDocumentOutput
from raganything.utils import separate_content


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_add_document(
        file_path: str,
        parse_method: Optional[str] = None,
        doc_id: Optional[str] = None,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
    ) -> AddDocumentOutput:
        """把一份文件加入 RAG 知識庫。

        Args:
            file_path: 絕對路徑或相對於 cwd 的檔案路徑。
            parse_method: 'auto' / 'ocr' / 'txt'，未指定走 config 預設。
            doc_id: 指定文件 ID；未指定時由 RAGAnything 依內容 hash 自動產生。
            split_by_character: 額外的分塊分隔字元。
            split_by_character_only: 若 True 則只用 character 分塊，不再做 token 分塊。

        Returns:
            AddDocumentOutput: 包含 doc_id、chunk_count、處理時間等。

        Raises:
            FileNotFoundError: file_path 不存在。
            RuntimeError: parser 安裝不完整或 Ollama 模型缺失（啟動自檢階段）。
        """
        start = time.time()
        rag = await rag_instance.get_rag()

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")

        effective_method = parse_method or rag.config.parse_method
        output_dir = rag.config.parser_output_dir

        content_list, content_doc_id = await rag.parse_document(
            str(path),
            output_dir=output_dir,
            parse_method=effective_method,
            display_stats=False,
        )
        final_doc_id = doc_id or content_doc_id
        file_name = rag._get_file_reference(str(path))

        text_content, multimodal_items = separate_content(content_list)

        if multimodal_items and hasattr(rag, "set_content_source_for_context"):
            rag.set_content_source_for_context(content_list, rag.config.content_format)

        if text_content.strip():
            await rag.lightrag.ainsert(
                input=text_content,
                ids=final_doc_id,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
            )

        if multimodal_items:
            await rag._process_multimodal_content(
                multimodal_items, file_name, final_doc_id
            )
        else:
            await rag._mark_multimodal_processing_complete(final_doc_id)

        chunk_count = 0
        try:
            doc_record = await rag.lightrag.doc_status.get_by_id(final_doc_id)
            if doc_record:
                chunk_count = int(doc_record.get("chunks_count", 0) or 0)
        except Exception:
            # doc_status 讀取失敗不影響主流程
            pass

        return AddDocumentOutput(
            doc_id=final_doc_id,
            file_name=file_name,
            content_blocks=len(content_list),
            text_chars=len(text_content),
            multimodal_items=len(multimodal_items),
            chunk_count=chunk_count,
            elapsed_seconds=round(time.time() - start, 2),
            parser_used=rag.config.parser,
        )
