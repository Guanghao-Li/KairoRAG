"""Build keyword and vector indexes from raw KairoRAG data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kairorag.config import KairoConfig
from kairorag.indexing.keyword_index import KeywordIndex
from kairorag.indexing.vector_store import VectorStore
from kairorag.ingestion.chunker import chunk_documents
from kairorag.ingestion.document_loader import load_knowledge_base


def build_indexes(
    input_dir: str | Path = KairoConfig.raw_data_dir,
    output_dir: str | Path = KairoConfig.index_dir,
    chunk_size: int = KairoConfig.chunk_size,
    overlap: int = KairoConfig.chunk_overlap,
) -> dict[str, int]:
    """Build and persist all local indexes."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    documents, jobs = load_knowledge_base(input_path)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    keyword_index = KeywordIndex()
    keyword_index.add_chunks(chunks)
    keyword_index.save(output_path / "keyword_index.pkl")

    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    vector_store.save(output_path / "vector_store.pkl")

    (output_path / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "documents.json").write_text(
        json.dumps([doc.to_dict() for doc in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "jobs.json").write_text(
        json.dumps([job.to_dict() for job in jobs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"documents": len(documents), "jobs": len(jobs), "chunks": len(chunks)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KairoRAG indexes")
    parser.add_argument("--input", default=str(KairoConfig.raw_data_dir))
    parser.add_argument("--output", default=str(KairoConfig.index_dir))
    parser.add_argument("--chunk-size", type=int, default=KairoConfig.chunk_size)
    parser.add_argument("--overlap", type=int, default=KairoConfig.chunk_overlap)
    args = parser.parse_args()

    stats = build_indexes(args.input, args.output, args.chunk_size, args.overlap)
    print(
        "Built indexes: "
        f"{stats['documents']} documents, {stats['jobs']} jobs, {stats['chunks']} chunks"
    )


if __name__ == "__main__":
    main()

