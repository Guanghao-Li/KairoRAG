"""Read full chunks after search."""

from __future__ import annotations

import json
from pathlib import Path

from kairorag.config import INDEX_DIR
from kairorag.ingestion.metadata import active_for_retrieval
from kairorag.schemas import ChunkRecord, ReadChunkResult


def _load_chunks(index_dir: str | Path = INDEX_DIR) -> list[ChunkRecord]:
    path = Path(index_dir) / "chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkRecord(**item) for item in data]


def chunk_read(
    chunk_id: str,
    window: int = 1,
    include_archived: bool = False,
    index_dir: str | Path = INDEX_DIR,
) -> ReadChunkResult:
    """Read a full chunk and optionally adjacent chunk text."""

    chunks = _load_chunks(index_dir)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    target = by_id.get(chunk_id)
    if not target:
        return ReadChunkResult(chunk_id=chunk_id, text="", metadata={"error": "chunk_not_found"})
    status = str(target.metadata.get("verification_status", "active"))
    if not active_for_retrieval(status, include_archived=include_archived):
        return ReadChunkResult(
            chunk_id=chunk_id,
            text="",
            metadata={**target.metadata, "error": "archived_chunk_requires_include_archived"},
        )

    siblings = [
        chunk
        for chunk in chunks
        if chunk.doc_id == target.doc_id
        and abs(chunk.chunk_index - target.chunk_index) <= max(0, window)
    ]
    previous = next((chunk.text for chunk in siblings if chunk.chunk_index == target.chunk_index - 1), None)
    next_chunk = next((chunk.text for chunk in siblings if chunk.chunk_index == target.chunk_index + 1), None)
    return ReadChunkResult(
        chunk_id=target.chunk_id,
        text=target.text,
        prev_chunk=previous,
        next_chunk=next_chunk,
        metadata=target.metadata,
    )


def chunk_read_trace(result: ReadChunkResult) -> dict[str, object]:
    return {
        "tool": "chunk_read",
        "chunk_id": result.chunk_id,
        "token_count": result.metadata.get("token_count") or len(result.text.split()),
        "has_text": bool(result.text),
    }

