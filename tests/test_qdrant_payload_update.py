import pytest
from types import SimpleNamespace
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.vectorstores import QdrantVectorStoreProvider


class FakeQdrantClient:
    def __init__(self):
        self.set_payload_calls = []

    def set_payload(self, *, collection_name, payload, points):
        self.set_payload_calls.append(
            {"collection_name": collection_name, "payload": payload, "points": points}
        )


class FakeQdrantClientWithRetrieve(FakeQdrantClient):
    def retrieve(self, *, collection_name, ids, with_payload):
        del collection_name, ids, with_payload
        return [
            SimpleNamespace(
                payload={
                    "chunk_id": "chunk-1",
                    "last_verified_at": "2026-06-12T01:00:00+00:00",
                }
            )
        ]


def _settings():
    return KairoCloudSettings(
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_qdrant_update_payload_allows_allowed_fields():
    client = FakeQdrantClient()
    provider = QdrantVectorStoreProvider(_settings(), client=client)

    provider.update_payload(
        ["chunk-1"],
        {"verification_status": "closed", "archived": True},
        allowed_fields=["verification_status", "archived"],
    )

    assert client.set_payload_calls[0]["payload"]["verification_status"] == "closed"
    assert client.set_payload_calls[0]["points"]


def test_qdrant_update_payload_rejects_non_allowed_field():
    provider = QdrantVectorStoreProvider(_settings(), client=FakeQdrantClient())

    with pytest.raises(KairoProviderError) as exc_info:
        provider.update_payload(["chunk-1"], {"company": "Kairo"}, allowed_fields=["archived"])

    assert "未授权字段" in str(exc_info.value)


def test_qdrant_update_payload_rejects_empty_chunk_ids():
    provider = QdrantVectorStoreProvider(_settings(), client=FakeQdrantClient())

    with pytest.raises(ValueError):
        provider.update_payload([], {"archived": True}, allowed_fields=["archived"])


def test_qdrant_update_payload_rejects_protected_fields():
    provider = QdrantVectorStoreProvider(_settings(), client=FakeQdrantClient())

    for field_name in ["text", "vector", "chunk_id", "doc_id"]:
        with pytest.raises(KairoProviderError):
            provider.update_payload(["chunk-1"], {field_name: "bad"}, allowed_fields=[field_name])


def test_qdrant_update_payload_optimistic_check_rejects_older_write():
    provider = QdrantVectorStoreProvider(_settings(), client=FakeQdrantClientWithRetrieve())

    with pytest.raises(KairoProviderError) as exc_info:
        provider.update_payload(
            ["chunk-1"],
            {"last_verified_at": "2026-06-12T00:00:00+00:00"},
            allowed_fields=["last_verified_at"],
        )

    assert "乐观检查失败" in str(exc_info.value)
