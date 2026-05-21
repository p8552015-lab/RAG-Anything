"""raganything.mcp — MCP stdio server wrapper for RAGAnything

對外只 export server 入口函式，避免 module import 觸發重型依賴。
"""

from raganything.mcp.server import main as main

__all__ = ["main"]
