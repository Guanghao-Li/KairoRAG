"""Groundedness helpers."""

from __future__ import annotations


def citations_from_read_chunks(citation_ids: list[str], read_chunk_ids: list[str]) -> bool:
    """Return true when all citations point to chunks that were actually read."""

    if not citation_ids:
        return False
    read = set(read_chunk_ids)
    return all(chunk_id in read for chunk_id in citation_ids)

