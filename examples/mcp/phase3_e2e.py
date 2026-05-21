"""phase3_e2e.py — Phase 3 進階 tools 端到端

假設 Phase 2 e2e 已跑過，working_dir 有資料；本腳本驗證：
1. rag_extract_content：對檔案做純抽取，回 content_list 分類統計
2. rag_query_multimodal：用 fake 表格當 context 問問題
3. rag_delete_document：list 拿 doc_id → delete → list 確認消失

如果你想完全獨立跑（不依賴 Phase 2 結果），手動先用 Claude / Cursor 加文件，
再跑這個腳本。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _make_sample_pdf(path: str) -> None:
    """用 reportlab 動態生成 PDF 給 extract_content 測試 (docling 需要 PDF)。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Phase 3 Multimodal RAG Test Document")
    c.setFont("Helvetica", 11)
    y = 760
    for line in [
        "Multimodal RAG components:",
        "  - Images: dispatched to a vision model for understanding",
        "  - Tables: extracted as markdown or LaTeX",
        "  - Equations: preserved as raw LaTeX",
        "",
        "Sample equation: F1 = 2 * (precision * recall) / (precision + recall)",
        "",
        "Sample table:",
        "  Model              Dim    Size",
        "  bge-m3            1024   1.2GB",
        "  nomic-embed-text   768   274MB",
    ]:
        c.drawString(50, y, line)
        y -= 18
    c.save()


FAKE_TABLE = """| 模型 | 維度 | 大小 |
| --- | --- | --- |
| bge-m3 | 1024 | 1.2GB |
| nomic-embed-text | 768 | 274MB |
"""


def _print(label: str, payload: object) -> None:
    print(f"\n=== {label} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


async def main() -> int:
    server_bin = shutil.which("raganything-mcp") or str(
        Path(sys.executable).parent / "raganything-mcp"
    )
    print(f"[phase3] spawning {server_bin}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        sample_path = f.name
    _make_sample_pdf(sample_path)
    print(f"[phase3] sample for extract: {sample_path}")

    async with stdio_client(StdioServerParameters(command=server_bin)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. extract_content (不入庫)
            r = await session.call_tool(
                "rag_extract_content",
                arguments={"file_path": sample_path, "types": ["text"]},
            )
            if r.isError:
                print(f"ERROR: extract failed: {r.content}", file=sys.stderr)
                return 1
            _print("rag_extract_content", json.loads(r.content[0].text))

            # 2. query_multimodal (用 fake 表格)
            r = await session.call_tool(
                "rag_query_multimodal",
                arguments={
                    "query": "從這張表格看，哪個 embedding 模型維度比較大？",
                    "multimodal_content": [
                        {
                            "type": "table",
                            "table_data": FAKE_TABLE,
                            "table_caption": "Embedding 模型比較",
                        }
                    ],
                    "mode": "hybrid",
                },
            )
            if r.isError:
                print(f"ERROR: query_multimodal failed: {r.content}", file=sys.stderr)
                return 2
            _print("rag_query_multimodal", json.loads(r.content[0].text))

            # 3. list → delete → list
            r = await session.call_tool("rag_list_documents", arguments={})
            before = json.loads(r.content[0].text)
            _print("rag_list_documents (before delete)", before)

            non_dup_docs = [
                d for d in before["documents"] if not d["doc_id"].startswith("dup-")
            ]
            if non_dup_docs:
                target = non_dup_docs[0]["doc_id"]
                r = await session.call_tool(
                    "rag_delete_document", arguments={"doc_id": target}
                )
                _print("rag_delete_document", json.loads(r.content[0].text))

                r = await session.call_tool("rag_list_documents", arguments={})
                after = json.loads(r.content[0].text)
                _print("rag_list_documents (after delete)", after)
                if any(d["doc_id"] == target for d in after["documents"]):
                    print(
                        f"ERROR: doc {target} still in list after delete",
                        file=sys.stderr,
                    )
                    return 3
            else:
                print("[phase3] skip delete: 沒有非 duplicate 文件可刪")

    print("\n[phase3] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
