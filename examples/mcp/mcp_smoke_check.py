"""test_client.py — Phase 1 驗證腳本

啟動 raganything-mcp stdio server，列出 tools 並呼叫 rag_get_status。
不依賴 mcp Inspector（不用裝 Node.js），純 Python SDK。
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    server_bin = shutil.which("raganything-mcp")
    if not server_bin:
        # fallback to venv path
        venv_bin = Path(sys.executable).parent / "raganything-mcp"
        if venv_bin.exists():
            server_bin = str(venv_bin)
    if not server_bin:
        print("ERROR: raganything-mcp not found in PATH", file=sys.stderr)
        return 1

    print(f"[client] spawning server: {server_bin}")
    params = StdioServerParameters(command=server_bin)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[client] initialize OK")

            tools = await session.list_tools()
            print(f"\n=== tools/list ({len(tools.tools)} tools) ===")
            for t in tools.tools:
                print(f"- {t.name}")
                if t.description:
                    print(f"    {t.description.splitlines()[0]}")

            print("\n=== call rag_get_status ===")
            result = await session.call_tool("rag_get_status", arguments={})
            if result.isError:
                print(f"ERROR: {result.content}", file=sys.stderr)
                return 2
            for content in result.content:
                text = getattr(content, "text", None)
                if text:
                    print(text)

    print("\n[client] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
