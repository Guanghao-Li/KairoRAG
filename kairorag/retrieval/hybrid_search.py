"""Hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairorag.config import INDEX_DIR
from kairorag.retrieval.keyword_search import keyword_search
from kairorag.retrieval.semantic_search import semantic_search
from kairorag.schemas import SearchResult


def hybrid_search(
    query: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    include_archived: bool = False,
    index_dir: str | Path = INDEX_DIR,
) -> list[SearchResult]:
    """Combine keyword and semantic results while deduplicating chunk IDs."""

    keyword_results = keyword_search(
        query, top_k=top_k, filters=filters, include_archived=include_archived, index_dir=index_dir
    )
    semantic_results = semantic_search(
        query, top_k=top_k, filters=filters, include_archived=include_archived, index_dir=index_dir
    )
    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for source_results, weight in ((keyword_results, 1.0), (semantic_results, 0.8)):
        for rank, result in enumerate(source_results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (60 + rank)
            if result.chunk_id not in fused or result.score > fused[result.chunk_id].score:
                fused[result.chunk_id] = result

    merged = []
    for chunk_id, result in fused.items():
        merged.append(
            SearchResult(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                source_type=result.source_type,
                title=result.title,
                snippet=result.snippet,
                metadata={**result.metadata, "hybrid_score": scores[chunk_id]},
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]

