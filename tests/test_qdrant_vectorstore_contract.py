from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.vectorstores import QdrantVectorStoreProvider, VectorChunk


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


class FakeQdrantClient:
    def __init__(self):
        self.created = None
        self.upserted = None
        self.search_request = None
        self.deleted = None

    def collection_exists(self, collection_name):
        del collection_name
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs

    def upsert(self, **kwargs):
        self.upserted = kwargs

    def search(self, **kwargs):
        self.search_request = kwargs
        return [
            SimpleNamespace(
                id="point-1",
                score=0.88,
                payload={
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "text": "Qdrant 证据文本",
                    "metadata": {"source_type": "job", "archived": False},
                },
            )
        ]

    def delete(self, **kwargs):
        self.deleted = kwargs

    def get_collections(self):
        return SimpleNamespace(collections=[])


class ExistingCollectionClient(FakeQdrantClient):
    def __init__(self, *, size=2, distance="Cosine"):
        super().__init__()
        self.size = size
        self.distance = distance
        self.deleted_collection = None

    def collection_exists(self, collection_name):
        del collection_name
        return True

    def get_collection(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.size, distance=self.distance))
            )
        )

    def delete_collection(self, **kwargs):
        self.deleted_collection = kwargs["collection_name"]


def test_qdrant_provider_collection_upsert_search_delete_and_healthcheck():
    client = FakeQdrantClient()
    provider = QdrantVectorStoreProvider(_settings(), client=client)

    provider.ensure_collection(vector_size=2)
    assert client.created["collection_name"] == "kairo_chunks"

    provider.upsert_chunks(
        [
            VectorChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="Qdrant 证据文本",
                vector=[0.1, 0.2],
                metadata={"source_type": "job", "archived": False},
            )
        ]
    )
    point = client.upserted["points"][0]
    assert point.payload["chunk_id"] == "chunk-1"
    assert point.payload["metadata"]["source_type"] == "job"

    results = provider.search([0.1, 0.2], top_k=3, filters={"archived": False, "source_type": "job"})
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.88
    assert client.search_request["query_filter"] is not None

    provider.delete_chunks(["chunk-1"])
    assert client.deleted["collection_name"] == "kairo_chunks"
    assert provider.healthcheck() is True


def test_qdrant_provider_existing_collection_matching_config_passes():
    client = ExistingCollectionClient(size=2, distance="Cosine")
    provider = QdrantVectorStoreProvider(_settings(), client=client)

    provider.ensure_collection(vector_size=2)

    assert client.created is None


def test_qdrant_provider_existing_collection_size_mismatch_errors():
    client = ExistingCollectionClient(size=3, distance="Cosine")
    provider = QdrantVectorStoreProvider(_settings(), client=client)

    with pytest.raises(KairoProviderError) as exc_info:
        provider.ensure_collection(vector_size=2)

    message = str(exc_info.value)
    assert "现有 vector size=3" in message
    assert "期望 vector size=2" in message


def test_qdrant_provider_existing_collection_distance_mismatch_errors():
    client = ExistingCollectionClient(size=2, distance="Dot")
    provider = QdrantVectorStoreProvider(_settings(qdrant_distance="Cosine"), client=client)

    with pytest.raises(KairoProviderError) as exc_info:
        provider.ensure_collection(vector_size=2)

    assert "现有 distance=Dot" in str(exc_info.value)
    assert "期望 distance=Cosine" in str(exc_info.value)


def test_qdrant_provider_recreates_collection_when_enabled():
    client = ExistingCollectionClient(size=3, distance="Dot")
    provider = QdrantVectorStoreProvider(_settings(cloud_recreate_collection=True), client=client)

    provider.ensure_collection(vector_size=2)

    assert client.deleted_collection == "kairo_chunks"
    assert client.created["collection_name"] == "kairo_chunks"
