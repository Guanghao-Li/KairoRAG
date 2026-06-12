import inspect

import pytest
from pydantic import SecretStr

from kairorag.cloud.freshness.verifier import CloudJobFreshnessVerifier
from kairorag.cloud.manifest import CloudChunkRecord, CloudIndexManifest, save_manifest
from kairorag.cloud.runtime import build_cloud_agent_runtime
from kairorag.config import KairoCloudSettings
from kairorag.config.settings import CloudRuntimeConfigurationError
from kairorag.providers.embeddings import EmbeddingResult
from kairorag.providers.keyword import BM25KeywordSearchProvider, KeywordDocument
from kairorag.providers.llm import LLMResponse
from kairorag.providers.rerankers import BaseScoreRerankerProvider


class FakeLLMProvider:
    model = "fake-llm"

    def complete(self, messages, *, temperature=0.0, response_format=None):
        return LLMResponse(content='{"answer":"ok","citations":[]}', model=self.model, usage={})

    def complete_with_tools(self, messages, tools, *, temperature=0.0):
        return LLMResponse(content="", model=self.model, usage={})


class FakeEmbeddingProvider:
    model = "fake-embedding"

    def embed_texts(self, texts):
        return [EmbeddingResult(text=text, vector=[1.0, 0.0], model=self.model, usage={}) for text in texts]

    def embed_query(self, query):
        return self.embed_texts([query])[0]


class FakeVectorStore:
    def search(self, query_vector, *, top_k, filters=None):
        return []


class FakeWebProvider:
    def search(self, query, *, max_results=5):
        return []


def _settings(tmp_path, **overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
        "cloud_index_manifest_path": str(tmp_path / "manifest.json"),
        "cloud_bm25_index_path": str(tmp_path / "bm25.json"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _patch_builders(monkeypatch):
    monkeypatch.setattr("kairorag.cloud.runtime.build_llm_provider", lambda settings: FakeLLMProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_embedding_provider", lambda settings: FakeEmbeddingProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_vector_store_provider", lambda settings: FakeVectorStore())
    monkeypatch.setattr("kairorag.cloud.runtime.build_keyword_search_provider", lambda settings: BM25KeywordSearchProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_reranker_provider", lambda settings: BaseScoreRerankerProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_web_search_provider", lambda settings: FakeWebProvider())


def _write_index(settings):
    manifest = CloudIndexManifest(
        created_at="2026-06-12T00:00:00+00:00",
        embedding_model="fake-embedding",
        vector_store_provider="qdrant",
        vector_collection="kairo_chunks",
        keyword_search_provider="bm25",
        chunk_count=1,
        chunks=[CloudChunkRecord("c1", "d1", "文档", "RAG BM25 完整文本", "company_doc", {})],
    )
    save_manifest(manifest, settings.cloud_index_manifest_path)
    provider = BM25KeywordSearchProvider()
    provider.index([KeywordDocument("c1", "d1", "文档", "RAG BM25", {"source_type": "company_doc"})])
    provider.save_keyword_documents(settings.cloud_bm25_index_path)


def test_freshness_enabled_runtime_builds_verifier(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path)
    _write_index(settings)

    runtime = build_cloud_agent_runtime(settings)

    assert runtime.web_search_provider is not None
    assert runtime.freshness_verifier is not None
    assert runtime.freshness_updater is not None


def test_freshness_disabled_runtime_skips_verifier(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path, tavily_api_key=None, job_freshness_enabled=False)
    _write_index(settings)

    runtime = build_cloud_agent_runtime(settings)

    assert runtime.web_search_provider is None
    assert runtime.freshness_verifier is None


def test_runtime_missing_key_fails_when_freshness_enabled(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path, tavily_api_key=None)
    _write_index(settings)

    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        build_cloud_agent_runtime(settings)

    assert "TAVILY_API_KEY" in str(exc_info.value)


def test_cloud_freshness_verifier_does_not_reference_legacy_modules():
    source = inspect.getsource(CloudJobFreshnessVerifier)

    assert "maintenance" not in source
    assert "job_verifier" not in source
    assert "mock" not in source.lower()
