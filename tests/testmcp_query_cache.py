import pytest

from raganything.mcp.cache import clear_query_response_cache


class FakeKVStorage:
    def __init__(self, data=None):
        self._data = data or {}
        self.index_done_calls = 0

    async def delete(self, ids):
        for item_id in ids:
            self._data.pop(item_id, None)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeLightRAG:
    def __init__(self):
        self.llm_response_cache = FakeKVStorage(
            {
                "hybrid:query:stale": {
                    "cache_type": "query",
                    "return": "no context",
                },
                "multimodal_query:stale": {
                    "cache_type": "multimodal_query",
                    "return": "old multimodal answer",
                },
                "hybrid:keywords:keep": {
                    "cache_type": "keywords",
                    "return": "{}",
                },
                "default:extract:keep": {
                    "cache_type": "extract",
                    "return": "entities",
                },
            }
        )


@pytest.mark.asyncio
async def test_clear_query_response_cache_only_removes_final_answers():
    lightrag = FakeLightRAG()

    deleted = await clear_query_response_cache(lightrag)

    assert deleted == 2
    assert set(lightrag.llm_response_cache._data) == {
        "hybrid:keywords:keep",
        "default:extract:keep",
    }
    assert lightrag.llm_response_cache.index_done_calls == 1
