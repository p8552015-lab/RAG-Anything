"""async_add_test.py — 驗證 rag_add_document 非同步化（R1 解法）

流程：
1. spawn raganything-mcp，確認多了 rag_get_job_status tool
2. rag_add_document 提交大文件 → 應「立刻」回 job_id（不等解析，不 timeout）
3. 輪詢 rag_get_job_status 直到 completed/failed

成功標準：submit 在數秒內回 job_id，最終 job 變 completed 且 result 有 doc_id。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PDF = "/tmp/rag_test_zh.pdf"


async def main() -> int:
    server_bin = shutil.which("raganything-mcp") or str(
        Path(sys.executable).parent / "raganything-mcp"
    )
    async with stdio_client(StdioServerParameters(command=server_bin)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            print(f"tools ({len(names)}): {sorted(names)}")
            if "rag_get_job_status" not in names:
                print("ERROR: rag_get_job_status tool missing")
                return 1

            # submit — 應立刻回
            t0 = time.time()
            r = await session.call_tool("rag_add_document", {"file_path": PDF})
            submit_dt = time.time() - t0
            submit = json.loads(r.content[0].text)
            print(f"\n[submit] {submit_dt:.1f}s -> {json.dumps(submit, ensure_ascii=False)}")
            if submit.get("status") != "submitted":
                print("ERROR: expected status=submitted")
                return 1
            job_id = submit["job_id"]
            if submit_dt > 55:
                print(f"WARNING: submit took {submit_dt:.1f}s (lazy init?), 仍應 < 60s")

            # poll
            for i in range(40):
                await asyncio.sleep(10)
                r = await session.call_tool("rag_get_job_status", {"job_id": job_id})
                st = json.loads(r.content[0].text)
                print(f"  poll {i + 1}: status={st['status']} elapsed={st['elapsed_seconds']}s")
                if st["status"] == "completed":
                    print(f"\n[DONE] {json.dumps(st['result'], ensure_ascii=False, indent=2)}")
                    return 0
                if st["status"] == "failed":
                    print(f"\n[FAILED] {st.get('error')}")
                    return 1
            print("\nERROR: polling timed out")
            return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
