"""raganything.mcp.server — stdio MCP server entry

由 ``raganything-mcp`` console script 啟動。流程：
1. ``_setup_env``：載入專案根目錄 ``.env``（如果存在）
2. ``_setup_path``：把 venv/bin 與 LibreOffice 加進 PATH（R10），這樣
   ``subprocess.run(["mineru", ...])`` 才找得到 CLI
3. ``_setup_logging``：log 走 stderr，stdio MCP 的 stdout 留給 JSON-RPC
4. 建立 ``FastMCP`` 並註冊 tool
5. ``mcp.run("stdio")``

Phase 1 只註冊 ``rag_get_status``，其他 tool 由 Phase 2/3 補上。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("raganything.mcp")


def _setup_env() -> None:
    """從 fork 根目錄載入 .env（如果存在），不覆寫已存在的環境變數。"""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)


def _setup_path() -> None:
    """補 PATH（R10）：venv/bin 與 LibreOffice 所在目錄。

    Claude Desktop spawn server 進程時 PATH 經常只有最小集合，導致
    raganything 內 ``subprocess.run(["mineru", ...])`` 找不到 CLI。
    """
    venv_bin = Path(sys.executable).parent
    libreoffice_dir = os.environ.get(
        "LIBREOFFICE_BIN_DIR", "/Applications/LibreOffice.app/Contents/MacOS"
    )
    parts = [str(venv_bin), libreoffice_dir, os.environ.get("PATH", "")]
    os.environ["PATH"] = ":".join(p for p in parts if p)


def _setup_logging() -> None:
    """所有 log 寫到 stderr；stdio MCP 的 stdout 留給 JSON-RPC。"""
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """raganything-mcp console script 入口。"""
    _setup_env()
    _setup_path()
    _setup_logging()

    # Lazy import：環境就緒後再 import 重型模組
    from mcp.server.fastmcp import FastMCP

    from raganything.mcp.tools import register_all

    mcp = FastMCP("rag-anything")
    register_all(mcp)

    logger.info("rag-anything MCP server starting on stdio …")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
