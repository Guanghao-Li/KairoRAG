"""向量数据库 provider 接口和 Qdrant 实现。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


SUPPORTED_FILTER_KEYS = {"source_type", "verification_status", "archived", "company", "job_id"}


@dataclass(frozen=True)
class VectorChunk:
    chunk_id: str
    doc_id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStoreProvider(Protocol):
    def ensure_collection(self, vector_size: int) -> None:
        ...

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        ...

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        ...

    def healthcheck(self) -> bool:
        ...


class QdrantVectorStoreProvider:
    """使用 Qdrant 的真实向量库 provider。"""

    def __init__(self, settings: KairoCloudSettings, client: Any | None = None) -> None:
        if not settings.qdrant_url:
            raise KairoProviderError("无法初始化 Qdrant provider：缺少 QDRANT_URL。")
        if not settings.qdrant_api_key or not settings.qdrant_api_key.get_secret_value().strip():
            raise KairoProviderError("无法初始化 Qdrant provider：缺少 QDRANT_API_KEY。")
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        if client is not None:
            self.client = client
            return
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover - 依赖缺失分支
            raise KairoProviderError("无法初始化 Qdrant provider：缺少 qdrant-client 依赖。") from exc
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            check_compatibility=False,
        )

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0。")
        try:
            models = _qdrant_models()
            exists = self._collection_exists()
            if exists:
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"Qdrant collection 初始化失败：{exc}") from exc

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        if not chunks:
            return
        try:
            models = _qdrant_models()
            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=chunk.vector,
                    payload=_payload_from_chunk(chunk),
                )
                for chunk in chunks
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"Qdrant upsert 失败：{exc}") from exc

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not query_vector:
            raise ValueError("query_vector 不能为空。")
        if top_k <= 0:
            return []
        try:
            query_filter = _build_filter(filters)
            if hasattr(self.client, "search"):
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                points = getattr(response, "points", response)
            return [_result_from_point(point) for point in points]
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"Qdrant search 失败：{exc}") from exc

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        try:
            models = _qdrant_models()
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[_point_id(chunk_id) for chunk_id in chunk_ids]),
            )
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"Qdrant delete 失败：{exc}") from exc

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))
        collections = self.client.get_collections()
        names = [getattr(collection, "name", "") for collection in getattr(collections, "collections", [])]
        return self.collection_name in names


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except Exception as exc:  # pragma: no cover - 依赖缺失分支
        raise KairoProviderError("无法使用 Qdrant provider：缺少 qdrant-client 依赖。") from exc
    return models


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kairorag:{chunk_id}"))


def _payload_from_chunk(chunk: VectorChunk) -> dict[str, Any]:
    return {
        **chunk.metadata,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def _build_filter(filters: dict[str, Any] | None) -> Any | None:
    filters = filters or {}
    conditions = []
    models = _qdrant_models()
    for key, value in filters.items():
        if key not in SUPPORTED_FILTER_KEYS or value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions) if conditions else None


def _result_from_point(point: Any) -> VectorSearchResult:
    payload = dict(getattr(point, "payload", {}) or {})
    metadata = dict(payload.get("metadata") or {})
    if not metadata:
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"chunk_id", "doc_id", "text"}
        }
    return VectorSearchResult(
        chunk_id=str(payload.get("chunk_id", getattr(point, "id", ""))),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        score=float(getattr(point, "score", 0.0)),
        metadata=metadata,
    )
