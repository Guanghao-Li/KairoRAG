"""Legacy / Deprecated：pickle 本地向量库，仅保留给离线 demo 和迁移对照。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from kairorag.indexing.embedding import HashingEmbeddingProvider, cosine_similarity
from kairorag.indexing.keyword_index import _make_snippet
from kairorag.ingestion.metadata import metadata_matches
from kairorag.schemas import ChunkRecord, SearchResult


class VectorStore:
    """Legacy / Deprecated：使用 hashing embedding 和 pickle 的本地向量索引。"""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.provider = HashingEmbeddingProvider(dim=dim)
        self.chunks: list[ChunkRecord] = []
        self.embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        for chunk in chunks:
            self.chunks.append(chunk)
            self.embeddings.append(self.provider.embed(f"{chunk.metadata.get('title', '')} {chunk.text}"))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_archived: bool = False,
    ) -> list[SearchResult]:
        query_vector = self.provider.embed(query)
        scored: list[SearchResult] = []
        query_terms = query.lower().split()
        for chunk, embedding in zip(self.chunks, self.embeddings):
            if not metadata_matches(chunk.metadata, filters, include_archived=include_archived):
                continue
            score = cosine_similarity(query_vector, embedding)
            if score <= 0:
                continue
            scored.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    source_type=chunk.source_type,
                    title=str(chunk.metadata.get("title", "")),
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
    def load(cls, path: str | Path) -> "VectorStore":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
