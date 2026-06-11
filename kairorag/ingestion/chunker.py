"""Document chunking with stable chunk IDs."""

from __future__ import annotations

import re
from typing import Iterable

from kairorag.ingestion.metadata import estimate_tokens
from kairorag.schemas import ChunkRecord, DocumentRecord


def _tokenish_units(text: str) -> list[str]:
    units = re.findall(r"[A-Za-z0-9_#+./:-]+|[\u4e00-\u9fff]|[^\s]", text)
    return units or [text]


def _join_units(units: list[str]) -> str:
    text = " ".join(units)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+([\u4e00-\u9fff])\s+", r"\1", text)
    return text.strip()


def chunk_documents(
    documents: Iterable[DocumentRecord],
    chunk_size: int = 120,
    overlap: int = 24,
) -> list[ChunkRecord]:
    """Split documents into stable overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[ChunkRecord] = []
    for doc in documents:
        units = _tokenish_units(doc.text)
        step = chunk_size - overlap
        index = 0
        for start in range(0, len(units), step):
            window = units[start : start + chunk_size]
            if not window:
                break
            text = _join_units(window)
            chunk_id = f"{doc.doc_id}_chunk_{index:03d}"
            metadata = {
                **doc.metadata,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_type": doc.source_type,
                "chunk_index": index,
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_type=doc.source_type,
                    text=text,
                    chunk_index=index,
                    metadata=metadata,
                    token_count=estimate_tokens(text),
                )
            )
            index += 1
            if start + chunk_size >= len(units):
                break
    return chunks

