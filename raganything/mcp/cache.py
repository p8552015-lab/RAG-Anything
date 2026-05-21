"""Cache helpers for MCP tools."""

from __future__ import annotations

from typing import Any

from raganything.mcp.cleanup import _call_index_done, _kv_data


async def clear_query_response_cache(lightrag: Any) -> int:
    """Invalidate final query-answer caches after knowledge-base mutations.

    LightRAG caches final query answers in ``llm_response_cache``. If a query is
    asked before a document is indexed, that no-context answer can otherwise be
    reused after the document is added. Keep keyword/extraction caches because
    they are not final corpus-grounded answers.
    """
    storage = getattr(lightrag, "llm_response_cache", None)
    data = _kv_data(storage)
    if not data:
        return 0

    keys_to_delete: list[str] = []
    for key, value in list(data.items()):
        cache_type = value.get("cache_type") if isinstance(value, dict) else None
        if cache_type in {"query", "multimodal_query"} or ":query:" in key:
            keys_to_delete.append(key)

    if not keys_to_delete:
        return 0

    await storage.delete(keys_to_delete)
    await _call_index_done(storage)
    return len(keys_to_delete)
