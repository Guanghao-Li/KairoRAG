import httpx
import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import LLMResponse
from kairorag.providers.rerankers import (
    BaseScoreRerankerProvider,
    CohereRerankProvider,
    CrossEncoderRerankerProvider,
    JinaRerankProvider,
    OpenAIListwiseRerankProvider,
    RerankCandidate,
    VoyageRerankProvider,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        return self.response


class FakeLLMProvider:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, messages, *, temperature=0.0, response_format=None):
        self.calls.append((messages, temperature, response_format))
        return LLMResponse(content=self.content, model="fake", usage={})

    def complete_with_tools(self, messages, tools, *, temperature=0.0):
        raise AssertionError("测试不应调用 tool calling。")


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.2, 0.9, 0.1][: len(pairs)]


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "cohere_api_key": SecretStr("placeholder-cohere"),
        "jina_api_key": SecretStr("placeholder-jina"),
        "voyage_api_key": SecretStr("placeholder-voyage"),
        "cross_encoder_model": "fake-cross-encoder",
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _candidates():
    return [
        RerankCandidate("c1", "第一段 RAG 文本", {"title": "一"}, 0.1),
        RerankCandidate("c2", "第二段 Qdrant 文本", {"title": "二"}, 0.9),
        RerankCandidate("c3", "第三段 BM25 文本", {"title": "三"}, 0.2),
    ]


def test_cohere_rerank_provider_parses_fake_response():
    client = FakeHTTPClient(FakeResponse({"results": [{"index": 1, "relevance_score": 0.88}]}))
    provider = CohereRerankProvider(_settings(), client=client)

    results = provider.rerank("RAG", _candidates(), top_k=1)

    assert results[0].chunk_id == "c2"
    assert results[0].metadata["title"] == "二"
    assert client.calls[0]["json"]["top_n"] == 1


def test_jina_rerank_provider_parses_fake_response():
    client = FakeHTTPClient(FakeResponse({"results": [{"index": 0, "relevance_score": 0.7}]}))
    provider = JinaRerankProvider(_settings(), client=client)

    results = provider.rerank("RAG", _candidates(), top_k=1)

    assert results[0].chunk_id == "c1"
    assert client.calls[0]["url"] == "https://api.jina.ai/v1/rerank"


def test_voyage_rerank_provider_accepts_data_field():
    client = FakeHTTPClient(FakeResponse({"data": [{"index": 2, "relevance_score": 0.66}]}))
    provider = VoyageRerankProvider(_settings(), client=client)

    results = provider.rerank("RAG", _candidates(), top_k=1)

    assert results[0].chunk_id == "c3"
    assert client.calls[0]["json"]["top_k"] == 1


def test_openai_listwise_rerank_provider_parses_json_and_sorts():
    llm = FakeLLMProvider(
        '{"results":[{"chunk_id":"c1","score":0.4,"reason":"相关"},'
        '{"chunk_id":"c2","score":0.9,"reason":"最相关"}]}'
    )
    provider = OpenAIListwiseRerankProvider(_settings(), llm_provider=llm)

    results = provider.rerank("RAG", _candidates(), top_k=2)

    assert [result.chunk_id for result in results] == ["c2", "c1"]
    assert results[0].metadata["rerank_reason"] == "最相关"


def test_openai_listwise_rerank_provider_rejects_invalid_json():
    provider = OpenAIListwiseRerankProvider(_settings(), llm_provider=FakeLLMProvider("不是 JSON"))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.rerank("RAG", _candidates(), top_k=1)

    assert "JSON" in str(exc_info.value)


def test_rerank_provider_http_500_does_not_leak_api_key():
    client = FakeHTTPClient(FakeResponse({"error": "bad"}, status_code=500))
    provider = CohereRerankProvider(_settings(cohere_api_key=SecretStr("sk-secret-cohere")), client=client)

    with pytest.raises(KairoProviderError) as exc_info:
        provider.rerank("RAG", _candidates(), top_k=1)

    assert "sk-secret-cohere" not in str(exc_info.value)
    assert "500" in str(exc_info.value)


def test_rerank_provider_timeout_is_provider_error():
    client = FakeHTTPClient(error=httpx.TimeoutException("timeout"))
    provider = JinaRerankProvider(_settings(), client=client)

    with pytest.raises(KairoProviderError) as exc_info:
        provider.rerank("RAG", _candidates(), top_k=1)

    assert "超时" in str(exc_info.value)


def test_rerank_input_validation_and_empty_candidates():
    provider = BaseScoreRerankerProvider()

    assert provider.rerank("RAG", [], top_k=1) == []
    with pytest.raises(ValueError):
        provider.rerank("", _candidates(), top_k=1)
    with pytest.raises(ValueError):
        provider.rerank("RAG", _candidates(), top_k=0)


def test_provider_unknown_index_is_ignored():
    client = FakeHTTPClient(FakeResponse({"results": [{"index": 99, "relevance_score": 0.9}]}))
    provider = CohereRerankProvider(_settings(), client=client)

    assert provider.rerank("RAG", _candidates(), top_k=1) == []


def test_cross_encoder_uses_fake_model_without_download():
    provider = CrossEncoderRerankerProvider(_settings(), model=FakeCrossEncoder())

    results = provider.rerank("RAG", _candidates(), top_k=2)

    assert [result.chunk_id for result in results] == ["c2", "c1"]
