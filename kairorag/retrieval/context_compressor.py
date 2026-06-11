"""Citation-preserving extractive context compression."""

from __future__ import annotations

import re

from kairorag.indexing.embedding import tokenize
from kairorag.schemas import ReadChunkResult


def compress_context(
    query: str,
    chunks: list[ReadChunkResult],
    max_sentences_per_chunk: int = 3,
) -> list[dict[str, str]]:
    """Keep query-relevant sentences while preserving chunk IDs."""

    query_terms = set(tokenize(query))
    blocks: list[dict[str, str]] = []
    for chunk in chunks:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+|\n+", chunk.text) if item.strip()]
        selected: list[str] = []
        for sentence in sentences:
            if query_terms & set(tokenize(sentence)):
                selected.append(sentence)
            if len(selected) >= max_sentences_per_chunk:
                break
        if not selected and sentences:
            selected = sentences[:1]
        if selected:
            blocks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "title": str(chunk.metadata.get("title", "")),
                    "source_type": str(chunk.metadata.get("source_type", "")),
                    "text": " ".join(selected),
                }
            )
    return blocks

