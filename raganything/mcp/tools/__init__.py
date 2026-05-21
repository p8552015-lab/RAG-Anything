"""raganything.mcp.tools — 集中 tool 註冊

每個 tool 一個檔案；server.py 呼叫 ``register_all(mcp)`` 完成綁定，
避免 server.py 自己掛太多 @mcp.tool() 函式。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    """把所有 tools 註冊到 MCP server。

    Phase 2 註冊：add_document / query / list_documents / status。
    Phase 3 擴充：delete_document / query_multimodal / extract_content。
    """
    from raganything.mcp.tools import add_document as _add
    from raganything.mcp.tools import delete_document as _del
    from raganything.mcp.tools import extract_content as _extract
    from raganything.mcp.tools import list_documents as _list
    from raganything.mcp.tools import query as _query
    from raganything.mcp.tools import query_multimodal as _query_mm
    from raganything.mcp.tools import status as _status

    _add.register(mcp)
    _del.register(mcp)
    _query.register(mcp)
    _query_mm.register(mcp)
    _list.register(mcp)
    _extract.register(mcp)
    _status.register(mcp)
