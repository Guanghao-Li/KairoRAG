from types import SimpleNamespace

from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.vectorstores import QdrantVectorStoreProvider, VectorChunk


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


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
