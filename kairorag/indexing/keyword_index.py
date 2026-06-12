"""Legacy / Deprecated：keyword overlap 本地索引，仅保留给离线 demo。"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from kairorag.indexing.embedding import tokenize
from kairorag.ingestion.metadata import metadata_matches
from kairorag.schemas import ChunkRecord, SearchResult


class KeywordIndex:
    """Legacy / Deprecated：基于词频重叠的轻量索引，不再是 cloud 主路径。"""

    def __init__(self) -> None:
        self.chunks: list[ChunkRecord] = []
        self.term_counts: dict[str, Counter[str]] = {}

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        self.chunks.extend(chunks)
        for chunk in chunks:
            enriched = f"{chunk.metadata.get('title', '')} {chunk.text}"
            self.term_counts[chunk.chunk_id] = Counter(tokenize(enriched))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_archived: bool = False,
    ) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_counter = Counter(query_terms)
        scored: list[SearchResult] = []
        for chunk in self.chunks:
            if not metadata_matches(chunk.metadata, filters, include_archived=include_archived):
                continue
            counts = self.term_counts.get(chunk.chunk_id, Counter())
            score = 0.0
            for term, q_count in query_counter.items():
                score += counts.get(term, 0) * (1.0 + min(len(term), 12) / 12.0) * q_count
            title = str(chunk.metadata.get("title", ""))
            title_lower = title.lower()
            for term in query_terms:
                if term in title_lower:
                    score += 2.0
            if score > 0:
                scored.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        score=score,
                        source_type=chunk.source_type,
                        title=title,
                        snippet=_make_snippet(chunk.text, query_terms),
                        metadata=chunk.metadata,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "KeywordIndex":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def _make_snippet(text: str, query_terms: list[str], width: int = 220) -> str:
    lowered = text.lower()
    first = min((lowered.find(term) for term in query_terms if term in lowered), default=0)
    start = max(0, first - width // 3)
    snippet = text[start : start + width].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + width < len(text):
        snippet += "..."
    return snippet
