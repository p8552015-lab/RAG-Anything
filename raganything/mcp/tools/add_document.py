"""rag_add_document — 把文件加入 RAG 知識庫（非同步，B 方案）

非同步化（R1）：Claude Code 對單次 MCP tool 呼叫有 ~60s 硬 timeout 且無法調整，
而大文件的 add（解析 + gemma4 抽實體）動輒數分鐘。因此 ``rag_add_document``
改成「提交即回 job_id」，實際處理在 MCP server 的 event loop 背景跑完，
再由 ``rag_get_job_status(job_id)`` 取結果。

解析流程仍走 B 方案（避開 R8 的 _upsert_doc_status regression）：
    parse_document → separate_content → lightrag.ainsert → _process_multimodal_content
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from raganything.mcp import jobs, rag_instance
from raganything.mcp.schemas import AddDocumentSubmitOutput
from raganything.utils import separate_content


def _text_only_multimodal_enabled() -> bool:
    raw = os.environ.get("RAGANYTHING_MCP_TEXT_ONLY_MULTIMODAL", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _multimodal_item_to_text(item: dict[str, Any]) -> str:
    """Best-effort text projection for parsed non-text blocks.

    This is intentionally opt-in via RAGANYTHING_MCP_TEXT_ONLY_MULTIMODAL. It is
    useful for code-heavy Markdown docs where the generic multimodal processor is
    slower than the value it adds, while still preserving code/table/equation
    content in the text index.
    """
    content_type = item.get("type", "unknown")
    parts: list[str] = []
    for key in (
        "text",
        "code_body",
        "table_body",
        "table_html",
        "latex",
        "equation",
        "content",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            joined = " ".join(str(v).strip() for v in value if str(v).strip())
            if joined:
                parts.append(joined)
    if not parts:
        return ""
    return f"[{content_type}]\n" + "\n".join(parts)


def _doc_status_value(doc_record: dict[str, Any]) -> str:
    status = doc_record.get("status", "")
    return getattr(status, "value", status)


def _doc_status_error(doc_record: dict[str, Any]) -> str:
    return (
        str(
            doc_record.get("error_msg")
            or doc_record.get("error")
            or doc_record.get("message")
            or ""
        )
        .strip()
    )


async def _raise_if_doc_failed(rag: Any, doc_id: str, stage: str) -> dict[str, Any] | None:
    """Raise when LightRAG persisted a failed doc_status without propagating.

    Some LightRAG ingestion paths persist ``status=failed`` but keep the outer
    coroutine alive, especially when text KG insertion fails before multimodal
    post-processing. The MCP job should surface that terminal failure promptly.
    """
    doc_record = await rag.lightrag.doc_status.get_by_id(doc_id)
    if not doc_record:
        return None

    status = _doc_status_value(doc_record)
    if status == "failed":
        error = _doc_status_error(doc_record) or "unknown ingestion error"
        chunk_count = doc_record.get("chunks_count", 0) or 0
        raise RuntimeError(
            f"Document ingestion failed during {stage}: {error} "
            f"(doc_id={doc_id}, chunks_count={chunk_count})"
        )
    return doc_record


async def _perform_add(
    rag: Any,
    path: Path,
    parse_method: Optional[str],
    doc_id: Optional[str],
    split_by_character: Optional[str],
    split_by_character_only: bool,
) -> dict[str, Any]:
    """實際解析 + 入庫，回 AddDocumentOutput 形狀的 dict。背景 job 執行此函式。"""
    start = time.time()
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
    if _text_only_multimodal_enabled() and multimodal_items:
        projected = [
            text for item in multimodal_items if (text := _multimodal_item_to_text(item))
        ]
        if projected:
            text_content = "\n\n".join(part for part in [text_content, *projected] if part)
        multimodal_items = []

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
        await _raise_if_doc_failed(rag, final_doc_id, "text insertion")

    if multimodal_items:
        await rag._process_multimodal_content(multimodal_items, file_name, final_doc_id)
        await _raise_if_doc_failed(rag, final_doc_id, "multimodal insertion")
    else:
        await rag._mark_multimodal_processing_complete(final_doc_id)
        await _raise_if_doc_failed(rag, final_doc_id, "completion")

    chunk_count = 0
    document_status = ""
    document_error = ""
    try:
        doc_record = await rag.lightrag.doc_status.get_by_id(final_doc_id)
        if doc_record:
            chunk_count = int(doc_record.get("chunks_count", 0) or 0)
            document_status = str(_doc_status_value(doc_record))
            document_error = _doc_status_error(doc_record)
    except Exception:
        pass

    return {
        "doc_id": final_doc_id,
        "file_name": file_name,
        "content_blocks": len(content_list),
        "text_chars": len(text_content),
        "multimodal_items": len(multimodal_items),
        "chunk_count": chunk_count,
        "document_status": document_status,
        "document_error": document_error,
        "elapsed_seconds": round(time.time() - start, 2),
        "parser_used": rag.config.parser,
    }


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_add_document(
        file_path: str,
        parse_method: Optional[str] = None,
        doc_id: Optional[str] = None,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
    ) -> AddDocumentSubmitOutput:
        """把文件加入 RAG 知識庫（非同步提交，立刻回 job_id）。

        實際解析 + 抽實體在背景跑，避開 MCP 60s timeout。用回傳的 job_id
        呼叫 ``rag_get_job_status(job_id)`` 查進度與結果。

        Args:
            file_path: 絕對路徑或相對於 cwd 的檔案路徑。
            parse_method: 'auto' / 'ocr' / 'txt'，未指定走 config 預設。
            doc_id: 指定文件 ID；未指定時由內容 hash 自動產生。
            split_by_character: 額外的分塊分隔字元。
            split_by_character_only: 若 True 只用 character 分塊。

        Returns:
            AddDocumentSubmitOutput: 含 job_id；用 rag_get_job_status 取最終結果。

        Raises:
            FileNotFoundError: file_path 不存在。
            RuntimeError: Ollama/parser 啟動自檢失敗。
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")

        # 確保 RAGAnything 已就緒（首次會跑啟動自檢，通常 <15s，在 timeout 內）
        rag = await rag_instance.get_rag()

        job = jobs.create_job("add_document", file_name=path.name)
        jobs.launch(
            job,
            lambda: _perform_add(
                rag,
                path,
                parse_method,
                doc_id,
                split_by_character,
                split_by_character_only,
            ),
        )

        return AddDocumentSubmitOutput(
            job_id=job.job_id,
            status="submitted",
            file_name=path.name,
            message=(
                "已提交背景處理；submitted 只代表已排入佇列，不代表入庫成功。用 rag_get_job_status('"
                + job.job_id
                + "') 查最終 completed/failed 狀態；大文件或多模態文件可能超過數分鐘。"
            ),
        )
