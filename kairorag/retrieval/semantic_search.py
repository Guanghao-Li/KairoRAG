"""Legacy / Deprecated：基于本地 hashing vectors 的语义检索工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairorag.config import INDEX_DIR
from kairorag.indexing.vector_store import VectorStore
from kairorag.schemas import SearchResult


def semantic_search(
    query: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    include_archived: bool = False,
    index_dir: str | Path = INDEX_DIR,
) -> list[SearchResult]:
    """Legacy：执行本地 hashing vector 相似度检索。"""

    filters = filters or {}
    include_archived = bool(filters.get("include_archived", include_archived))
    store = VectorStore.load(Path(index_dir) / "vector_store.pkl")
    return store.search(query, top_k=top_k, filters=filters, include_archived=include_archived)


def semantic_search_trace(query: str, results: list[SearchResult]) -> dict[str, Any]:
    return {
        "tool": "semantic_search",
        "query": query,
        "result_count": len(results),
        "chunk_ids": [result.chunk_id for result in results],
    }
