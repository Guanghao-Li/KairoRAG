"""Simple deterministic reranking."""

from __future__ import annotations

from kairorag.indexing.embedding import tokenize
from kairorag.schemas import SearchResult


def rerank_context(query: str, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
    """Rerank candidates using query/snippet overlap plus original score."""

    query_terms = set(tokenize(query))
    reranked: list[SearchResult] = []
    for result in results:
        snippet_terms = set(tokenize(f"{result.title} {result.snippet}"))
        overlap = len(query_terms & snippet_terms)
        score = result.score + overlap * 0.15
        if result.metadata.get("verification_status") == "stale":
            score *= 0.75
        reranked.append(
            SearchResult(
                chunk_id=result.chunk_id,
                score=score,
                source_type=result.source_type,
                title=result.title,
                snippet=result.snippet,
                metadata={**result.metadata, "rerank_overlap": overlap},
            )
        )
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k] if top_k else reranked

