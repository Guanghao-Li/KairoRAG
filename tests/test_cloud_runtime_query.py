import pytest
from pydantic import SecretStr

from kairorag.cloud.manifest import CloudChunkRecord, CloudIndexManifest, save_manifest
from kairorag.cloud.runtime import build_cloud_agent_runtime, build_cloud_query_runtime
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
    def ensure_collection(self, vector_size):
        del vector_size

    def upsert_chunks(self, chunks):
        del chunks

    def search(self, query_vector, *, top_k, filters=None):
        return []

    def delete_chunks(self, chunk_ids):
        del chunk_ids

    def healthcheck(self):
        return True


def _settings(tmp_path):
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
        cloud_index_manifest_path=str(tmp_path / "manifest.json"),
        cloud_bm25_index_path=str(tmp_path / "bm25.json"),
    )


def _patch_builders(monkeypatch):
    monkeypatch.setattr("kairorag.cloud.runtime.build_llm_provider", lambda settings: FakeLLMProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_embedding_provider", lambda settings: FakeEmbeddingProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_vector_store_provider", lambda settings: FakeVectorStore())
    monkeypatch.setattr("kairorag.cloud.runtime.build_keyword_search_provider", lambda settings: BM25KeywordSearchProvider())
    monkeypatch.setattr("kairorag.cloud.runtime.build_reranker_provider", lambda settings: BaseScoreRerankerProvider())


def test_build_cloud_query_runtime_errors_when_manifest_missing(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)

    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        build_cloud_query_runtime(_settings(tmp_path))

    assert "python -m kairorag.cloud.index" in str(exc_info.value)


def test_build_cloud_query_runtime_errors_when_bm25_missing(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path)
    save_manifest(_manifest(), settings.cloud_index_manifest_path)

    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        build_cloud_query_runtime(settings)

    assert "缺少 cloud BM25 index" in str(exc_info.value)


def test_build_cloud_query_runtime_loads_manifest_and_bm25(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path)
    save_manifest(_manifest(), settings.cloud_index_manifest_path)
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "文档", "RAG BM25", {"source_type": "company_doc"}),
            KeywordDocument("c2", "d2", "其他", "invoice workflow", {"source_type": "company_doc"}),
            KeywordDocument("c3", "d3", "更多", "travel policy", {"source_type": "company_doc"}),
        ]
    )
    provider.save_keyword_documents(settings.cloud_bm25_index_path)

    runtime = build_cloud_query_runtime(settings)

    assert runtime.manifest.chunk_count == 1
    assert runtime.keyword_search.search("RAG", top_k=1)[0].chunk_id == "c1"
    assert runtime.query_service is not None


def test_build_cloud_agent_runtime_adds_llm_agent(monkeypatch, tmp_path):
    _patch_builders(monkeypatch)
    settings = _settings(tmp_path)
    save_manifest(_manifest(), settings.cloud_index_manifest_path)
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "文档", "RAG BM25", {"source_type": "company_doc"}),
            KeywordDocument("c2", "d2", "其他", "invoice workflow", {"source_type": "company_doc"}),
            KeywordDocument("c3", "d3", "更多", "travel policy", {"source_type": "company_doc"}),
        ]
    )
    provider.save_keyword_documents(settings.cloud_bm25_index_path)

    runtime = build_cloud_agent_runtime(settings)

    assert runtime.llm_agent is not None
    assert runtime.manifest.chunk_count == 1


def _manifest():
    return CloudIndexManifest(
        created_at="2026-06-12T00:00:00+00:00",
        embedding_model="fake-embedding",
        vector_store_provider="qdrant",
        vector_collection="kairo_chunks",
        keyword_search_provider="bm25",
        chunk_count=1,
        chunks=[CloudChunkRecord("c1", "d1", "文档", "RAG BM25 完整文本", "company_doc", {})],
    )
