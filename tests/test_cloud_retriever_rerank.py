from pydantic import SecretStr

from kairorag.cloud.manifest import CloudChunkRecord, CloudIndexManifest
from kairorag.cloud.retriever import CloudRetriever
from kairorag.config import KairoCloudSettings
from kairorag.providers.embeddings import EmbeddingResult
from kairorag.providers.keyword import BM25KeywordSearchProvider, KeywordDocument
from kairorag.providers.rerankers import BaseScoreRerankerProvider, RerankResult
from kairorag.providers.vectorstores import VectorSearchResult


class FakeEmbeddingProvider:
    model = "fake"

    def embed_texts(self, texts):
        return [EmbeddingResult(text=text, vector=[1.0], model=self.model, usage={}) for text in texts]

    def embed_query(self, query):
        return EmbeddingResult(text=query, vector=[1.0], model=self.model, usage={})


class FakeVectorStore:
    def search(self, query_vector, *, top_k, filters=None):
        return [
            VectorSearchResult("c1", "d1", "第一段", 0.9, {"title": "一"}),
            VectorSearchResult("c2", "d2", "第二段", 0.8, {"title": "二"}),
        ][:top_k]

    def ensure_collection(self, vector_size):
        pass

    def upsert_chunks(self, chunks):
        pass

    def delete_chunks(self, chunk_ids):
        pass

    def update_payload(self, chunk_ids, payload_patch, *, allowed_fields):
        pass

    def healthcheck(self):
        return True


class ReorderingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, *, top_k):
        self.calls.append((query, [candidate.chunk_id for candidate in candidates], top_k))
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        return [
            RerankResult("c2", 0.99, dict(by_id["c2"].metadata)),
            RerankResult("c1", 0.5, dict(by_id["c1"].metadata)),
        ][:top_k]


class EmptyReranker:
    def rerank(self, query, candidates, *, top_k):
        return []


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
        "rerank_top_k": 2,
        "rerank_candidate_count": 4,
        "max_search_results": 4,
        "max_chunks_to_read": 2,
        "observability_enabled": False,
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _keyword_provider():
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "一", "第一段 RAG", {}),
            KeywordDocument("c2", "d2", "二", "第二段 Qdrant", {}),
        ]
    )
    return provider


def _manifest():
    return CloudIndexManifest(
        created_at="2026-06-12T00:00:00+00:00",
        embedding_model="fake",
        vector_store_provider="qdrant",
        vector_collection="kairo_chunks",
        keyword_search_provider="bm25",
        chunk_count=2,
        chunks=[
            CloudChunkRecord("c1", "d1", "一", "第一段完整文本", "company_doc", {}),
            CloudChunkRecord("c2", "d2", "二", "第二段完整文本", "company_doc", {}),
        ],
    )


def _retriever(settings, reranker):
    return CloudRetriever(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(),
        _keyword_provider(),
        reranker,
        _manifest(),
    )


def test_retrieve_calls_reranker_and_uses_reranked_chunk_read_order():
    reranker = ReorderingReranker()
    result = _retriever(_settings(), reranker).retrieve("RAG Qdrant")

    assert reranker.calls
    assert [chunk.chunk_id for chunk in result.read_chunks] == ["c2", "c1"]


def test_retrieve_trace_and_metrics_include_rerank_detail():
    result = _retriever(_settings(), ReorderingReranker()).retrieve("RAG Qdrant")

    rerank_step = next(step for step in result.trace if step.step == "rerank")
    assert rerank_step.detail["reranker_provider"] == "base_score"
    assert rerank_step.detail["rerank_candidate_count"] >= 1
    assert result.metrics["rerank_enabled"] is True
    assert result.metrics["rerank_output_count"] == 2


def test_retrieve_cloud_reranker_empty_result_is_explicit_without_fallback():
    settings = _settings(reranker_provider="cohere", cohere_api_key=SecretStr("placeholder-cohere"))

    result = _retriever(settings, EmptyReranker()).retrieve("RAG Qdrant")

    assert result.read_chunks == []
    assert result.metrics["rerank_output_count"] == 0
    assert "空结果" in result.metrics["rerank_error"]


def test_base_score_baseline_still_works():
    result = _retriever(_settings(), BaseScoreRerankerProvider()).retrieve("RAG Qdrant")

    assert result.read_chunks
    assert result.metrics["reranker_provider"] == "base_score"
