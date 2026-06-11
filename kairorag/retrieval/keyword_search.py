"""Keyword retrieval tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairorag.config import INDEX_DIR
from kairorag.indexing.keyword_index import KeywordIndex
from kairorag.schemas import SearchResult


def keyword_search(
    query: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    include_archived: bool = False,
    index_dir: str | Path = INDEX_DIR,
) -> list[SearchResult]:
    """Search by exact lexical overlap, excluding closed/duplicate jobs by default."""

    filters = filters or {}
    include_archived = bool(filters.get("include_archived", include_archived))
    index = KeywordIndex.load(Path(index_dir) / "keyword_index.pkl")
    return index.search(query, top_k=top_k, filters=filters, include_archived=include_archived)


def keyword_search_trace(query: str, results: list[SearchResult]) -> dict[str, Any]:
    return {
        "tool": "keyword_search",
        "query": query,
        "result_count": len(results),
        "chunk_ids": [result.chunk_id for result in results],
    }

