"""向量数据库 provider 接口和 Qdrant 实现。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


SUPPORTED_FILTER_KEYS = {"source_type", "verification_status", "archived", "company", "job_id"}
PROTECTED_PAYLOAD_FIELDS = {"text", "vector", "chunk_id", "doc_id"}


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

    def update_payload(
        self,
        chunk_ids: list[str],
        payload_patch: dict[str, Any],
        *,
        allowed_fields: list[str],
    ) -> None:
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
            exists = self._collection_exists()
            if exists:
                if self.settings.cloud_recreate_collection:
                    self._recreate_collection(vector_size)
                    return
                current_size, current_distance = self._read_collection_vector_config()
                expected_distance = _normalize_distance(self.settings.qdrant_distance)
                if current_size != vector_size or current_distance != expected_distance:
                    raise KairoProviderError(
                        "Qdrant collection 配置不匹配："
                        f"collection={self.collection_name}，"
                        f"现有 vector size={current_size}，期望 vector size={vector_size}，"
                        f"现有 distance={current_distance or '未知'}，期望 distance={expected_distance}。"
                        "如需重建 collection，请设置 CLOUD_RECREATE_COLLECTION=true。"
                    )
                return
            self._create_collection(vector_size)
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

    def update_payload(
        self,
        chunk_ids: list[str],
        payload_patch: dict[str, Any],
        *,
        allowed_fields: list[str],
    ) -> None:
        """安全更新 Qdrant payload，只写入 freshness metadata 白名单字段。"""

        if not chunk_ids:
            raise ValueError("chunk_ids 不能为空。")
        patch = dict(payload_patch or {})
        if not patch:
            return
        allowed = set(allowed_fields or [])
        protected = PROTECTED_PAYLOAD_FIELDS & set(patch)
        if protected:
            raise KairoProviderError("Qdrant payload 安全写回拒绝更新受保护字段：" + "、".join(sorted(protected)))
        illegal = set(patch) - allowed
        if illegal:
            raise KairoProviderError("Qdrant payload 安全写回包含未授权字段：" + "、".join(sorted(illegal)))
        try:
            self._optimistic_payload_check(chunk_ids, patch)
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=patch,
                points=[_point_id(chunk_id) for chunk_id in chunk_ids],
            )
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"Qdrant payload 写回失败：{exc}") from exc

    def _optimistic_payload_check(self, chunk_ids: list[str], patch: dict[str, Any]) -> None:
        """在 client 支持 retrieve 时，避免旧 verification 覆盖更新 payload。"""

        if not hasattr(self.client, "retrieve"):
            return
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[_point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
        by_chunk_id = {str(chunk_id): False for chunk_id in chunk_ids}
        incoming_verified_at = patch.get("last_verified_at")
        for point in points or []:
            payload = dict(getattr(point, "payload", {}) or {})
            payload_chunk_id = str(payload.get("chunk_id") or "")
            if payload_chunk_id in by_chunk_id:
                by_chunk_id[payload_chunk_id] = True
            existing_verified_at = payload.get("last_verified_at")
            if existing_verified_at and incoming_verified_at and str(existing_verified_at) > str(incoming_verified_at):
                raise KairoProviderError(
                    "Qdrant payload 乐观检查失败：现有 last_verified_at 晚于本次写回，已拒绝覆盖。"
                )
        missing = [chunk_id for chunk_id, seen in by_chunk_id.items() if not seen]
        if points and missing:
            raise KairoProviderError("Qdrant payload 乐观检查失败：retrieve 结果缺少 chunk_id：" + "、".join(missing))

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

    def _create_collection(self, vector_size: int) -> None:
        models = _qdrant_models()
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=_qdrant_distance_model(models, self.settings.qdrant_distance),
            ),
        )

    def _recreate_collection(self, vector_size: int) -> None:
        if not hasattr(self.client, "delete_collection"):
            raise KairoProviderError("Qdrant client 不支持 delete_collection，无法按 CLOUD_RECREATE_COLLECTION 重建。")
        self.client.delete_collection(collection_name=self.collection_name)
        self._create_collection(vector_size)

    def _read_collection_vector_config(self) -> tuple[int | None, str | None]:
        if not hasattr(self.client, "get_collection"):
            raise KairoProviderError("Qdrant collection 已存在，但 client 不支持 get_collection，无法校验向量配置。")
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors = _find_vectors_config(info)
        size = _get_value(vectors, "size")
        distance = _get_value(vectors, "distance")
        if size is None and isinstance(vectors, dict) and vectors:
            first_vector = next(iter(vectors.values()))
            size = _get_value(first_vector, "size")
            distance = _get_value(first_vector, "distance")
        return int(size) if size is not None else None, _normalize_distance(distance)


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except Exception as exc:  # pragma: no cover - 依赖缺失分支
        raise KairoProviderError("无法使用 Qdrant provider：缺少 qdrant-client 依赖。") from exc
    return models


def _qdrant_distance_model(models: Any, distance: Any) -> Any:
    name = _distance_enum_name(distance)
    try:
        return getattr(models.Distance, name)
    except AttributeError as exc:
        raise KairoProviderError(f"不支持的 QDRANT_DISTANCE：{distance}") from exc


def _distance_enum_name(distance: Any) -> str:
    normalized = _normalize_distance(distance)
    mapping = {
        "cosine": "COSINE",
        "dot": "DOT",
        "euclid": "EUCLID",
        "manhattan": "MANHATTAN",
    }
    key = (normalized or "").lower()
    if key not in mapping:
        raise KairoProviderError(f"不支持的 QDRANT_DISTANCE：{distance}")
    return mapping[key]


def _normalize_distance(distance: Any) -> str | None:
    if distance is None:
        return None
    value = getattr(distance, "value", distance)
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == text and "_" not in text:
        text = text.lower()
    mapping = {
        "cosine": "Cosine",
        "dot": "Dot",
        "euclid": "Euclid",
        "euclidean": "Euclid",
        "manhattan": "Manhattan",
        "COSINE": "Cosine",
        "DOT": "Dot",
        "EUCLID": "Euclid",
        "MANHATTAN": "Manhattan",
    }
    return mapping.get(text, text[:1].upper() + text[1:])


def _find_vectors_config(info: Any) -> Any:
    candidates = [
        ("config", "params", "vectors"),
        ("config", "params", "vectors_config"),
        ("config", "vectors"),
        ("vectors_config",),
        ("params", "vectors"),
        ("vectors",),
    ]
    for path in candidates:
        value = info
        for key in path:
            value = _get_value(value, key)
            if value is None:
                break
        if value is not None:
            return value
    return None


def _get_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


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
