import pytest

from raganything.mcp.cleanup import cleanup_document_residue


class FakeKVStorage:
    def __init__(self, data=None):
        self._data = data or {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self._data.get(key)

    async def upsert(self, data):
        self._data.update(data)

    async def delete(self, ids):
        for item_id in ids:
            self._data.pop(item_id, None)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeVectorStorage:
    def __init__(self, data=None):
        self.storage = {"data": data or []}
        self.deleted = []
        self.index_done_calls = 0

    @property
    async def client_storage(self):
        return self.storage

    async def delete(self, ids):
        ids = set(ids)
        self.deleted.extend(ids)
        self.storage["data"] = [
            item for item in self.storage["data"] if item["__id__"] not in ids
        ]

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeGraph:
    def __init__(self):
        self.nodes = {
            "資安業者": {"id": "資安業者", "source_id": "chunk-orphan"},
            "共用實體": {"id": "共用實體", "source_id": "chunk-orphan<SEP>chunk-live"},
        }
        self.edges = [
            {
                "source": "資安業者",
                "target": "資安經費",
                "source_id": "chunk-orphan",
            }
        ]
        self.removed_nodes = []
        self.removed_edges = []
        self.upserted_nodes = []
        self.upserted_edges = []
        self.index_done_calls = 0

    async def get_all_nodes(self):
        return list(self.nodes.values())

    async def get_all_edges(self):
        return list(self.edges)

    async def remove_nodes(self, nodes):
        self.removed_nodes.extend(nodes)
        for node in nodes:
            self.nodes.pop(node, None)

    async def remove_edges(self, edges):
        self.removed_edges.extend(edges)
        edge_set = set(edges)
        self.edges = [
            edge
            for edge in self.edges
            if (edge["source"], edge["target"]) not in edge_set
            and (edge["target"], edge["source"]) not in edge_set
        ]

    async def upsert_node(self, node_id, node_data):
        self.upserted_nodes.append((node_id, node_data))
        self.nodes[node_id] = {"id": node_id, **node_data}

    async def upsert_edge(self, source, target, edge_data):
        self.upserted_edges.append((source, target, edge_data))

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeLightRAG:
    def __init__(self):
        self.text_chunks = FakeKVStorage(
            {
                "chunk-orphan": {
                    "full_doc_id": "doc-test",
                    "content": "residual chunk",
                },
                "chunk-live": {
                    "full_doc_id": "doc-other",
                    "content": "other chunk",
                },
            }
        )
        self.chunks_vdb = FakeVectorStorage(
            [
                {"__id__": "chunk-orphan", "full_doc_id": "doc-test"},
                {"__id__": "chunk-live", "full_doc_id": "doc-other"},
            ]
        )
        self.entities_vdb = FakeVectorStorage(
            [
                {"__id__": "ent-orphan", "source_id": "chunk-orphan"},
                {"__id__": "ent-live", "source_id": "chunk-live"},
            ]
        )
        self.relationships_vdb = FakeVectorStorage(
            [
                {"__id__": "rel-orphan", "source_id": "chunk-orphan"},
                {"__id__": "rel-live", "source_id": "chunk-live"},
            ]
        )
        self.entity_chunks = FakeKVStorage(
            {
                "資安業者": {"chunk_ids": ["chunk-orphan"], "count": 1},
                "共用實體": {"chunk_ids": ["chunk-orphan", "chunk-live"], "count": 2},
            }
        )
        self.relation_chunks = FakeKVStorage(
            {
                "資安業者<SEP>資安經費": {
                    "chunk_ids": ["chunk-orphan"],
                    "count": 1,
                }
            }
        )
        self.full_docs = FakeKVStorage({"doc-test": {"content": "doc"}})
        self.full_entities = FakeKVStorage({"doc-test": {"entity_names": ["資安業者"]}})
        self.full_relations = FakeKVStorage({"doc-test": {"relation_pairs": []}})
        self.doc_status = FakeKVStorage({"doc-test": {"status": "failed"}})
        self.chunk_entity_relation_graph = FakeGraph()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class FakeRAG:
    def __init__(self):
        self.lightrag = FakeLightRAG()


@pytest.mark.asyncio
async def test_cleanup_document_residue_removes_orphan_records_by_doc_id():
    rag = FakeRAG()

    summary = await cleanup_document_residue(rag, "doc-test")

    assert summary["chunk_ids_found"] == 1
    assert "chunk-orphan" not in rag.lightrag.text_chunks._data
    assert rag.lightrag.chunks_vdb.storage["data"] == [
        {"__id__": "chunk-live", "full_doc_id": "doc-other"}
    ]
    assert rag.lightrag.entities_vdb.deleted == ["ent-orphan"]
    assert rag.lightrag.relationships_vdb.deleted == ["rel-orphan"]
    assert "資安業者" not in rag.lightrag.entity_chunks._data
    assert rag.lightrag.entity_chunks._data["共用實體"]["chunk_ids"] == ["chunk-live"]
    assert rag.lightrag.relation_chunks._data == {}
    assert "doc-test" not in rag.lightrag.doc_status._data
    assert "資安業者" in rag.lightrag.chunk_entity_relation_graph.removed_nodes
    assert rag.lightrag.chunk_entity_relation_graph.nodes["共用實體"]["source_id"] == (
        "chunk-live"
    )


@pytest.mark.asyncio
async def test_cleanup_document_residue_can_preserve_doc_records_for_native_delete():
    rag = FakeRAG()

    summary = await cleanup_document_residue(
        rag,
        "doc-test",
        delete_doc_records=False,
    )

    assert summary["chunk_ids_found"] == 1
    assert summary["doc_records_deleted"] == 0
    assert "chunk-orphan" not in rag.lightrag.text_chunks._data
    assert "資安業者" in rag.lightrag.chunk_entity_relation_graph.removed_nodes
    assert rag.lightrag.doc_status._data["doc-test"]["status"] == "failed"

    post_summary = await cleanup_document_residue(rag, "doc-test")

    assert post_summary["doc_records_deleted"] == 4
    assert "doc-test" not in rag.lightrag.doc_status._data
