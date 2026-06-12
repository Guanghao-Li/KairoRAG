"""Cloud 索引 manifest 的读写工具。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CloudChunkRecord:
    """可由 chunk_read 直接读取的完整 chunk 记录。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CloudIndexManifest:
    """Cloud 索引的可读元数据，不保存 embedding 向量。"""

    created_at: str
    embedding_model: str
    vector_store_provider: str
    vector_collection: str
    keyword_search_provider: str
    chunk_count: int
    chunks: list[CloudChunkRecord]


def save_manifest(manifest: CloudIndexManifest, path: str | Path) -> None:
    """把 manifest 保存成缩进 JSON。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> CloudIndexManifest:
    """从 JSON 文件加载 cloud index manifest。"""

    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = [
        CloudChunkRecord(
            chunk_id=str(item["chunk_id"]),
            doc_id=str(item["doc_id"]),
            title=str(item.get("title", "")),
            text=str(item.get("text", "")),
            source_type=str(item.get("source_type", "")),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in payload.get("chunks", [])
    ]
    return CloudIndexManifest(
        created_at=str(payload.get("created_at", "")),
        embedding_model=str(payload.get("embedding_model", "")),
        vector_store_provider=str(payload.get("vector_store_provider", "")),
        vector_collection=str(payload.get("vector_collection", "")),
        keyword_search_provider=str(payload.get("keyword_search_provider", "")),
        chunk_count=int(payload.get("chunk_count", len(chunks))),
        chunks=chunks,
    )


def find_chunk(manifest: CloudIndexManifest, chunk_id: str) -> CloudChunkRecord | None:
    """按 chunk_id 查找 manifest 中的完整 chunk。"""

    for chunk in manifest.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    return None
