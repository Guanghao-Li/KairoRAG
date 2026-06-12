"""Cloud-native 索引构建流程。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from kairorag.cloud.manifest import CloudChunkRecord, CloudIndexManifest, save_manifest
from kairorag.config import RAW_DATA_DIR, KairoCloudSettings
from kairorag.ingestion.chunker import chunk_documents
from kairorag.ingestion.document_loader import load_documents
from kairorag.providers.embeddings import EmbeddingProvider
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.keyword import KeywordDocument, KeywordSearchProvider
from kairorag.providers.vectorstores import VectorChunk, VectorStoreProvider
from kairorag.schemas import ChunkRecord, utc_now


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudIndexStats:
    """Cloud 索引构建统计。"""

    documents_loaded: int
    chunks_created: int
    embeddings_created: int
    qdrant_upserted: int
    bm25_documents_indexed: int
    manifest_path: str
    bm25_index_path: str


class CloudIndexer:
    """读取知识库、生成 embedding，并写入 Qdrant 与 BM25。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        keyword_search: KeywordSearchProvider,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.keyword_search = keyword_search

    def build(self, *, recreate: bool = False) -> CloudIndexStats:
        """构建 cloud 主路径索引，不生成 legacy pickle 文件。"""

        logger.info("开始构建 cloud 索引。")
        documents = load_documents(RAW_DATA_DIR)
        if not documents:
            raise KairoProviderError("未加载到任何知识库文档，无法构建 cloud 索引。")

        chunks = chunk_documents(documents)
        if not chunks:
            raise KairoProviderError("知识库文档未生成任何 chunk，无法构建 cloud 索引。")

        manifest_chunks = [_manifest_chunk(chunk) for chunk in chunks]
        keyword_documents = [_keyword_document(chunk) for chunk in chunks]

        embeddings_created = 0
        qdrant_upserted = 0
        vector_size: int | None = None
        old_recreate = self.settings.cloud_recreate_collection
        if recreate:
            self.settings.cloud_recreate_collection = True
        try:
            for batch in _batched(chunks, self.settings.cloud_chunk_batch_size):
                embedding_results = self.embedding_provider.embed_texts([chunk.text for chunk in batch])
                if len(embedding_results) != len(batch):
                    raise KairoProviderError(
                        "embedding provider 返回数量不匹配："
                        f"expected={len(batch)}，actual={len(embedding_results)}。"
                    )
                if not embedding_results:
                    continue
                if vector_size is None:
                    vector_size = len(embedding_results[0].vector)
                    self.vector_store.ensure_collection(vector_size)
                vector_chunks: list[VectorChunk] = []
                for chunk, embedding in zip(batch, embedding_results):
                    if len(embedding.vector) != vector_size:
                        raise KairoProviderError(
                            f"chunk {chunk.chunk_id} 的 embedding 维度为 {len(embedding.vector)}，"
                            f"与首次维度 {vector_size} 不一致。"
                        )
                    vector_chunks.append(
                        VectorChunk(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            text=chunk.text,
                            vector=embedding.vector,
                            metadata=dict(chunk.metadata),
                        )
                    )
                self.vector_store.upsert_chunks(vector_chunks)
                embeddings_created += len(embedding_results)
                qdrant_upserted += len(vector_chunks)
        finally:
            if recreate:
                self.settings.cloud_recreate_collection = old_recreate

        self.keyword_search.index(keyword_documents)
        _save_keyword_documents(keyword_documents, self.settings.cloud_bm25_index_path)

        manifest = CloudIndexManifest(
            created_at=utc_now(),
            embedding_model=getattr(self.embedding_provider, "model", self.settings.openai_embedding_model),
            vector_store_provider=self.settings.vector_store_provider,
            vector_collection=self.settings.qdrant_collection,
            keyword_search_provider=self.settings.keyword_search_provider,
            chunk_count=len(manifest_chunks),
            chunks=manifest_chunks,
        )
        save_manifest(manifest, self.settings.cloud_index_manifest_path)
        logger.info(
            "cloud 索引构建完成：文档 %s，chunk %s，embedding %s。",
            len(documents),
            len(chunks),
            embeddings_created,
        )
        return CloudIndexStats(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            embeddings_created=embeddings_created,
            qdrant_upserted=qdrant_upserted,
            bm25_documents_indexed=len(keyword_documents),
            manifest_path=self.settings.cloud_index_manifest_path,
            bm25_index_path=self.settings.cloud_bm25_index_path,
        )


def _manifest_chunk(chunk: ChunkRecord) -> CloudChunkRecord:
    metadata = dict(chunk.metadata)
    return CloudChunkRecord(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=str(metadata.get("title", chunk.doc_id)),
        text=chunk.text,
        source_type=chunk.source_type,
        metadata=metadata,
    )


def _keyword_document(chunk: ChunkRecord) -> KeywordDocument:
    metadata = dict(chunk.metadata)
    return KeywordDocument(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=str(metadata.get("title", chunk.doc_id)),
        text=chunk.text,
        metadata=metadata,
    )


def _save_keyword_documents(documents: list[KeywordDocument], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(document) for document in documents]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _batched(items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    if batch_size <= 0:
        raise ValueError("cloud_chunk_batch_size 必须大于 0。")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
