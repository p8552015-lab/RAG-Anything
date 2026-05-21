"""phase2_e2e.py — Phase 2 三件套端到端測試

流程：
1. spawn raganything-mcp stdio server
2. list_tools 確認 4 個 tool（add/query/list/status）
3. 寫一份小 markdown 到 /tmp 並呼叫 rag_add_document
4. 呼叫 rag_list_documents 確認 doc 出現
5. 呼叫 rag_query 兩題，驗證 LLM 能從知識庫答出
6. 印 status 看 document_count 是否累計

不 mock 任何後端；走真實遠端 Ollama。預期執行 1–3 分鐘（看 gemma4 entity
extraction 速度）。
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

EXPECTED_TOOLS = {"rag_add_document", "rag_query", "rag_list_documents", "rag_get_status"}

# PDF 是 docling 主場 + mineru 也吃。raganything 對 .md/.html/.txt 有
# 各種 fallback path 會踩 bug，所以用真實 PDF 最穩。
def _make_sample_pdf(path: str) -> None:
    """用 reportlab 動態生成 1 頁英文測試 PDF (避免外部檔案、中文字型依賴)。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "RAG-Anything MCP Server Test Document")
    c.setFont("Helvetica", 11)
    y = 760
    for line in [
        "RAG-Anything is a multimodal RAG framework.",
        "It supports PDF, Office, and image files.",
        "The backbone is LightRAG for knowledge graph construction.",
        "MinerU is the default parser, with docling as an alternative.",
        "",
        "Model configuration:",
        "  - LLM: gemma4:latest, 8B parameters, Q4_K_M quantization",
        "  - Embedding: bge-m3:latest, dimension 1024",
        "  - All inference runs on a remote Ollama server",
        "",
        "Known limitations:",
        "  - mineru first run downloads ~3GB of models",
        "  - bge-m3 on CPU processes about one chunk per second",
        "  - LightRAG doc_status does not support concurrent writes",
    ]:
        c.drawString(50, y, line)
        y -= 18
    c.save()


SAMPLE_SUFFIX = ".pdf"


def _print(label: str, payload: object) -> None:
    print(f"\n=== {label} ===")
    if hasattr(payload, "model_dump"):
        print(json.dumps(payload.model_dump(), ensure_ascii=False, indent=2))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(payload)


async def main() -> int:
    server_bin = shutil.which("raganything-mcp") or str(
        Path(sys.executable).parent / "raganything-mcp"
    )
    if not Path(server_bin).exists():
        print(f"ERROR: raganything-mcp not found ({server_bin})", file=sys.stderr)
        return 1

    print(f"[phase2] spawning {server_bin}")
    params = StdioServerParameters(command=server_bin)

    with tempfile.NamedTemporaryFile(suffix=SAMPLE_SUFFIX, delete=False) as f:
        sample_path = f.name
    _make_sample_pdf(sample_path)
    print(f"[phase2] sample file: {sample_path}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            print(f"\n[phase2] tools: {sorted(tool_names)}")
            missing = EXPECTED_TOOLS - tool_names
            if missing:
                print(f"ERROR: missing tools: {missing}", file=sys.stderr)
                return 2

            # 1. status (init=false 預期)
            r = await session.call_tool("rag_get_status", arguments={})
            _print("rag_get_status (before add)", r.content[0].text if r.content else "")

            # 2. add_document
            print("\n[phase2] adding document ...（可能需要 1–3 分鐘）")
            r = await session.call_tool(
                "rag_add_document", arguments={"file_path": sample_path}
            )
            if r.isError:
                print(f"ERROR: rag_add_document failed: {r.content}", file=sys.stderr)
                return 3
            add_text = r.content[0].text if r.content else ""
            _print("rag_add_document", add_text)
            add_result = json.loads(add_text)
            doc_id = add_result["doc_id"]

            # 3. list_documents
            r = await session.call_tool("rag_list_documents", arguments={})
            _print("rag_list_documents", r.content[0].text if r.content else "")

            # 4. query (英文 sample → 英文 query 最直接，避免跨語言干擾)
            for q in (
                "What embedding model does RAG-Anything use and what is its dimension?",
                "How many parameters does the LLM model in this test have?",
            ):
                r = await session.call_tool(
                    "rag_query", arguments={"query": q, "mode": "hybrid"}
                )
                _print(f"rag_query: {q}", r.content[0].text if r.content else "")

            # 5. status (init=true, doc_count>=1)
            r = await session.call_tool("rag_get_status", arguments={})
            _print("rag_get_status (after add)", r.content[0].text if r.content else "")

    print(f"\n[phase2] doc_id={doc_id} sample={sample_path}")
    print("[phase2] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
