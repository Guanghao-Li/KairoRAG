from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.embeddings import OpenAIEmbeddingProvider


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


class FakeEmbeddings:
    def __init__(self, extra_rows=0):
        self.payload = None
        self.extra_rows = extra_rows

    def create(self, **payload):
        self.payload = payload
        row_count = len(payload["input"]) + self.extra_rows
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(index + 1)])
            for index in range(row_count)
        ]
        return SimpleNamespace(
            data=data,
            model=payload["model"],
            usage=SimpleNamespace(total_tokens=len(payload["input"])),
        )


class FakeClient:
    def __init__(self, extra_rows=0):
        self.embeddings = FakeEmbeddings(extra_rows=extra_rows)


def test_openai_embedding_provider_supports_multiple_texts():
    client = FakeClient()
    provider = OpenAIEmbeddingProvider(_settings(), client=client)

    results = provider.embed_texts(["第一段", "第二段"])

    assert [result.text for result in results] == ["第一段", "第二段"]
    assert results[1].vector == [1.0, 2.0]
    assert client.embeddings.payload["model"] == "text-embedding-3-small"


def test_openai_embedding_provider_supports_query_and_empty_list():
    provider = OpenAIEmbeddingProvider(_settings(), client=FakeClient())

    assert provider.embed_texts([]) == []
    query = provider.embed_query("检索问题")

    assert query.text == "检索问题"
    assert query.vector == [0.0, 1.0]


def test_openai_embedding_provider_rejects_blank_text():
    provider = OpenAIEmbeddingProvider(_settings(), client=FakeClient())

    with pytest.raises(ValueError):
        provider.embed_texts(["有效", "  "])


def test_openai_embedding_provider_rejects_missing_rows():
    provider = OpenAIEmbeddingProvider(_settings(), client=FakeClient(extra_rows=-1))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.embed_texts(["第一段", "第二段"])

    assert "expected=2" in str(exc_info.value)
    assert "actual=1" in str(exc_info.value)


def test_openai_embedding_provider_rejects_extra_rows():
    provider = OpenAIEmbeddingProvider(_settings(), client=FakeClient(extra_rows=1))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.embed_texts(["第一段", "第二段"])

    assert "expected=2" in str(exc_info.value)
    assert "actual=3" in str(exc_info.value)
