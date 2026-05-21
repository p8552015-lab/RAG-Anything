"""Best-effort cleanup helpers for MCP document deletion.

LightRAG's native ``adelete_by_doc_id`` depends on ``doc_status.chunks_list`` and
``full_entities/full_relations``. Failed partial ingestions can leave chunks and
graph references with ``full_doc_id`` / ``source_id`` links that are not present
in those registries. These helpers scan the local storage objects and remove
only records that are exclusively tied to the target document's chunk IDs.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from lightrag.utils import GRAPH_FIELD_SEP


def _split_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(GRAPH_FIELD_SEP) if part]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            result.extend(_split_source_ids(item))
        return result
    return [str(value)]


def _join_source_ids(values: Iterable[str]) -> str:
    return GRAPH_FIELD_SEP.join(value for value in values if value)


def _kv_data(storage: Any) -> dict[str, Any] | None:
    data = getattr(storage, "_data", None)
    if data is None:
        return None
    if hasattr(data, "_getvalue"):
        data = data._getvalue()
    return data


async def _vector_data(storage: Any) -> list[dict[str, Any]]:
    client_storage = getattr(storage, "client_storage", None)
    if client_storage is None:
        return []
    if inspect.isawaitable(client_storage):
        client_storage = await client_storage
    if not isinstance(client_storage, dict):
        return []
    data = client_storage.get("data", [])
    return data if isinstance(data, list) else []


async def _call_index_done(storage: Any) -> None:
    callback = getattr(storage, "index_done_callback", None)
    if callback is None:
        return
    result = callback()
    if inspect.isawaitable(result):
        await result


async def _delete_kv_records(storage: Any, ids: Iterable[str]) -> int:
    ids_list = list(dict.fromkeys(ids))
    if not ids_list:
        return 0
    await storage.delete(ids_list)
    return len(ids_list)


async def _delete_vector_records(storage: Any, ids: Iterable[str]) -> int:
    ids_list = list(dict.fromkeys(ids))
    if not ids_list:
        return 0
    await storage.delete(ids_list)
    return len(ids_list)


async def _collect_doc_chunk_ids(lightrag: Any, doc_id: str) -> set[str]:
    chunk_ids: set[str] = set()

    doc_status = getattr(lightrag, "doc_status", None)
    if doc_status is not None:
        record = await doc_status.get_by_id(doc_id)
        if record:
            chunk_ids.update(_split_source_ids(record.get("chunks_list", [])))

    text_chunks = _kv_data(getattr(lightrag, "text_chunks", None))
    if text_chunks:
        for chunk_id, chunk in text_chunks.items():
            if isinstance(chunk, dict) and chunk.get("full_doc_id") == doc_id:
                chunk_ids.add(chunk_id)

    chunks_vdb = getattr(lightrag, "chunks_vdb", None)
    if chunks_vdb is not None:
        for chunk in await _vector_data(chunks_vdb):
            if chunk.get("full_doc_id") == doc_id and chunk.get("__id__"):
                chunk_ids.add(chunk["__id__"])

    return chunk_ids


async def _cleanup_chunk_tracking_storage(
    storage: Any,
    chunk_ids: set[str],
) -> tuple[int, int]:
    data = _kv_data(storage)
    if not data:
        return 0, 0

    delete_keys: list[str] = []
    update_payload: dict[str, dict[str, Any]] = {}

    for key, value in list(data.items()):
        if not isinstance(value, dict):
            continue
        current_chunk_ids = _split_source_ids(value.get("chunk_ids", []))
        if not set(current_chunk_ids) & chunk_ids:
            continue

        remaining = [chunk_id for chunk_id in current_chunk_ids if chunk_id not in chunk_ids]
        if remaining:
            update_payload[key] = {
                **value,
                "chunk_ids": remaining,
                "count": len(remaining),
            }
        else:
            delete_keys.append(key)

    if update_payload:
        await storage.upsert(update_payload)
    deleted = await _delete_kv_records(storage, delete_keys)
    if update_payload or deleted:
        await _call_index_done(storage)
    return deleted, len(update_payload)


async def _cleanup_vector_by_source(storage: Any, chunk_ids: set[str]) -> int:
    ids_to_delete: list[str] = []
    for item in await _vector_data(storage):
        vector_id = item.get("__id__")
        source_ids = set(_split_source_ids(item.get("source_id")))
        if vector_id and source_ids and source_ids <= chunk_ids:
            ids_to_delete.append(vector_id)

    deleted = await _delete_vector_records(storage, ids_to_delete)
    if deleted:
        await _call_index_done(storage)
    return deleted


async def _cleanup_graph(graph: Any, chunk_ids: set[str]) -> tuple[int, int, int, int]:
    nodes_deleted = 0
    nodes_updated = 0
    edges_deleted = 0
    edges_updated = 0

    if not graph:
        return nodes_deleted, nodes_updated, edges_deleted, edges_updated

    if hasattr(graph, "get_all_nodes"):
        nodes_to_delete: list[str] = []
        for node in await graph.get_all_nodes():
            node_id = node.get("id") or node.get("entity_id")
            source_ids = _split_source_ids(node.get("source_id"))
            if not node_id or not source_ids:
                continue
            source_set = set(source_ids)
            if source_set <= chunk_ids:
                nodes_to_delete.append(node_id)
            elif source_set & chunk_ids:
                remaining = [source_id for source_id in source_ids if source_id not in chunk_ids]
                updated = {k: v for k, v in node.items() if k != "id"}
                updated["source_id"] = _join_source_ids(remaining)
                await graph.upsert_node(node_id, updated)
                nodes_updated += 1
        if nodes_to_delete:
            await graph.remove_nodes(nodes_to_delete)
            nodes_deleted = len(nodes_to_delete)

    if hasattr(graph, "get_all_edges"):
        edges_to_delete: list[tuple[str, str]] = []
        for edge in await graph.get_all_edges():
            source = edge.get("source")
            target = edge.get("target")
            source_ids = _split_source_ids(edge.get("source_id"))
            if not source or not target or not source_ids:
                continue
            source_set = set(source_ids)
            if source_set <= chunk_ids:
                edges_to_delete.append((source, target))
            elif source_set & chunk_ids:
                remaining = [source_id for source_id in source_ids if source_id not in chunk_ids]
                updated = {
                    k: v for k, v in edge.items() if k not in {"source", "target"}
                }
                updated["source_id"] = _join_source_ids(remaining)
                await graph.upsert_edge(source, target, updated)
                edges_updated += 1
        if edges_to_delete:
            await graph.remove_edges(edges_to_delete)
            edges_deleted = len(edges_to_delete)

    if nodes_deleted or nodes_updated or edges_deleted or edges_updated:
        await _call_index_done(graph)

    return nodes_deleted, nodes_updated, edges_deleted, edges_updated


async def cleanup_document_residue(
    rag: Any,
    doc_id: str,
    known_chunk_ids: Iterable[str] | None = None,
    *,
    delete_doc_records: bool = True,
) -> dict[str, int]:
    """Remove storage records that are exclusively tied to ``doc_id``.

    The function is deliberately conservative: entity/relation records are only
    deleted when all their source chunks belong to the target doc. Mixed-source
    records are retained and their source tracking is trimmed where possible.
    """
    lightrag = rag.lightrag
    chunk_ids = set(known_chunk_ids or [])
    chunk_ids.update(await _collect_doc_chunk_ids(lightrag, doc_id))

    summary = {
        "chunk_ids_found": len(chunk_ids),
        "text_chunks_deleted": 0,
        "chunk_vectors_deleted": 0,
        "entity_vectors_deleted": 0,
        "relationship_vectors_deleted": 0,
        "entity_chunk_records_deleted": 0,
        "entity_chunk_records_updated": 0,
        "relation_chunk_records_deleted": 0,
        "relation_chunk_records_updated": 0,
        "graph_nodes_deleted": 0,
        "graph_nodes_updated": 0,
        "graph_edges_deleted": 0,
        "graph_edges_updated": 0,
        "doc_records_deleted": 0,
    }

    if chunk_ids:
        summary["chunk_vectors_deleted"] = await _delete_vector_records(
            lightrag.chunks_vdb, sorted(chunk_ids)
        )
        summary["text_chunks_deleted"] = await _delete_kv_records(
            lightrag.text_chunks, sorted(chunk_ids)
        )

        summary["entity_vectors_deleted"] = await _cleanup_vector_by_source(
            lightrag.entities_vdb, chunk_ids
        )
        summary["relationship_vectors_deleted"] = await _cleanup_vector_by_source(
            lightrag.relationships_vdb, chunk_ids
        )

        (
            summary["entity_chunk_records_deleted"],
            summary["entity_chunk_records_updated"],
        ) = await _cleanup_chunk_tracking_storage(lightrag.entity_chunks, chunk_ids)
        (
            summary["relation_chunk_records_deleted"],
            summary["relation_chunk_records_updated"],
        ) = await _cleanup_chunk_tracking_storage(lightrag.relation_chunks, chunk_ids)

        (
            summary["graph_nodes_deleted"],
            summary["graph_nodes_updated"],
            summary["graph_edges_deleted"],
            summary["graph_edges_updated"],
        ) = await _cleanup_graph(lightrag.chunk_entity_relation_graph, chunk_ids)

        await _call_index_done(lightrag.chunks_vdb)
        await _call_index_done(lightrag.text_chunks)

    if delete_doc_records:
        for storage_name in ("full_docs", "full_entities", "full_relations", "doc_status"):
            storage = getattr(lightrag, storage_name, None)
            if storage is None:
                continue
            before = 0
            data = _kv_data(storage)
            if data is not None and doc_id in data:
                before = 1
            await storage.delete([doc_id])
            if before:
                summary["doc_records_deleted"] += 1
                await _call_index_done(storage)

    insert_done = getattr(lightrag, "_insert_done", None)
    if insert_done is not None:
        result = insert_done()
        if inspect.isawaitable(result):
            await result

    return summary
