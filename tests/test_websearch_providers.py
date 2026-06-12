import httpx
import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.websearch import BingSearchProvider, SerpAPISearchProvider, TavilySearchProvider


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeClient:
    payload = {}
    status_code = 200
    raised = None
    calls = []

    def __init__(self, *, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, *, json):
        self.calls.append(("post", url, json, self.timeout))
        if self.raised:
            raise self.raised
        return FakeResponse(self.payload, self.status_code)

    def get(self, url, *, params, headers=None):
        self.calls.append(("get", url, params, headers, self.timeout))
        if self.raised:
            raise self.raised
        return FakeResponse(self.payload, self.status_code)


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
        "serpapi_api_key": SecretStr("placeholder-serpapi"),
        "bing_api_key": SecretStr("placeholder-bing"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _patch_client(monkeypatch, payload, status_code=200, raised=None):
    FakeClient.payload = payload
    FakeClient.status_code = status_code
    FakeClient.raised = raised
    FakeClient.calls = []
    monkeypatch.setattr("kairorag.providers.websearch.httpx.Client", FakeClient)
    return FakeClient


def test_tavily_response_maps_to_web_search_result(monkeypatch):
    client = _patch_client(
        monkeypatch,
        {
            "results": [
                {
                    "title": "RAG Engineer",
                    "url": "https://jobs.example/rag",
                    "content": "Apply now",
                    "published_date": "2026-06-01",
                    "api_key": "placeholder-tavily",
                }
            ]
        },
    )

    results = TavilySearchProvider(_settings()).search("rag", max_results=1)

    assert results[0].title == "RAG Engineer"
    assert results[0].source == "tavily"
    assert "api_key" not in results[0].raw
    assert client.calls[0][0] == "post"


def test_serpapi_response_maps_to_web_search_result(monkeypatch):
    _patch_client(
        monkeypatch,
        {"organic_results": [{"title": "Job", "link": "https://example/jobs/1", "snippet": "Job details"}]},
    )

    results = SerpAPISearchProvider(_settings()).search("job", max_results=1)

    assert results[0].url == "https://example/jobs/1"
    assert results[0].source == "serpapi"


def test_bing_response_maps_to_web_search_result(monkeypatch):
    _patch_client(
        monkeypatch,
        {"webPages": {"value": [{"name": "Job", "url": "https://example/jobs/1", "snippet": "Apply now"}]}},
    )

    results = BingSearchProvider(_settings()).search("job", max_results=1)

    assert results[0].title == "Job"
    assert results[0].source == "bing"


def test_web_search_http_error_becomes_provider_error(monkeypatch):
    _patch_client(monkeypatch, {"error": "bad"}, status_code=500)

    with pytest.raises(KairoProviderError) as exc_info:
        TavilySearchProvider(_settings()).search("rag")

    assert "HTTP 500" in str(exc_info.value)
    assert "placeholder-tavily" not in str(exc_info.value)


def test_web_search_timeout_becomes_provider_error(monkeypatch):
    _patch_client(monkeypatch, {}, raised=httpx.TimeoutException("timeout"))

    with pytest.raises(KairoProviderError) as exc_info:
        SerpAPISearchProvider(_settings()).search("rag")

    assert "超时" in str(exc_info.value)


def test_empty_provider_results_return_empty_list(monkeypatch):
    _patch_client(monkeypatch, {"results": []})

    assert TavilySearchProvider(_settings()).search("rag") == []


def test_web_search_rejects_empty_query_and_invalid_max_results():
    provider = TavilySearchProvider(_settings())

    with pytest.raises(ValueError):
        provider.search("  ")
    with pytest.raises(ValueError):
        provider.search("rag", max_results=0)
