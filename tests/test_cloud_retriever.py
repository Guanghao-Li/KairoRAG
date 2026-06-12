from pydantic import SecretStr

from kairorag.cloud.manifest import CloudChunkRecord, CloudIndexManifest
from kairorag.cloud.retriever import CloudRetriever
from kairorag.config import KairoCloudSettings
from kairorag.providers.embeddings import EmbeddingResult
from kairorag.providers.keyword import BM25KeywordSearchProvider, KeywordDocument
from kairorag.providers.rerankers import BaseScoreRerankerProvider
from kairorag.providers.vectorstores import VectorSearchResult


class FakeEmbeddingProvider:
    model = "fake-embedding"

    def __init__(self):
        self.query_calls = []

    def embed_texts(self, texts):
        return [EmbeddingResult(text=text, vector=[1.0, 0.0], model=self.model, usage={}) for text in texts]

    def embed_query(self, query):
        self.query_calls.append(query)
        return EmbeddingResult(text=query, vector=[1.0, 0.0], model=self.model, usage={})


class FakeVectorStore:
    def __init__(self):
        self.search_calls = []

    def ensure_collection(self, vector_size):
        del vector_size

    def upsert_chunks(self, chunks):
        del chunks

    def search(self, query_vector, *, top_k, filters=None):
        self.search_calls.append((query_vector, top_k, filters))
        return [
            VectorSearchResult(
                chunk_id="c1",
                doc_id="d1",
                text="RAG 和 Qdrant 语义检索",
                score=0.9,
                metadata={"title": "语义文档", "source_type": "company_doc"},
            ),
            VectorSearchResult(
                chunk_id="c2",
                doc_id="d2",
                text="BM25 keyword retrieval",
                score=0.8,
                metadata={"title": "关键词文档", "source_type": "company_doc"},
            ),
        ][:top_k]

    def delete_chunks(self, chunk_ids):
        del chunk_ids

    def healthcheck(self):
        return True


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        max_search_results=5,
        max_chunks_to_read=2,
        max_context_tokens=1000,
    )


def _manifest():
    return CloudIndexManifest(
        created_at="2026-06-12T00:00:00+00:00",
        embedding_model="fake-embedding",
        vector_store_provider="qdrant",
        vector_collection="kairo_chunks",
        keyword_search_provider="bm25",
        chunk_count=2,
        chunks=[
            CloudChunkRecord("c1", "d1", "语义文档", "RAG 和 Qdrant 语义检索完整文本", "company_doc", {}),
            CloudChunkRecord("c2", "d2", "关键词文档", "BM25 keyword retrieval 完整文本", "company_doc", {}),
        ],
    )


def _keyword_provider():
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "语义文档", "RAG Qdrant", {"source_type": "company_doc"}),
            KeywordDocument("c2", "d2", "关键词文档", "BM25 keyword retrieval", {"source_type": "company_doc"}),
            KeywordDocument("c3", "d3", "其他文档", "invoice workflow", {"source_type": "company_doc"}),
        ]
    )
    return provider


def _retriever():
    return CloudRetriever(
        _settings(),
        FakeEmbeddingProvider(),
        FakeVectorStore(),
        _keyword_provider(),
        BaseScoreRerankerProvider(),
        _manifest(),
    )


def test_cloud_retriever_semantic_and_keyword_search():
    retriever = _retriever()

    semantic = retriever.semantic_search("RAG", top_k=2)
    keyword = retriever.keyword_search("BM25", top_k=2)

    assert semantic[0].source == "semantic"
    assert keyword[0].source == "keyword"


def test_cloud_retriever_hybrid_search_uses_rrf_and_deduplicates():
    retriever = _retriever()

    results = retriever.hybrid_search("RAG BM25", top_k=5)

    assert len({result.chunk_id for result in results}) == len(results)
    assert all(result.source == "hybrid" for result in results)
    assert "rrf_score" in results[0].metadata


def test_cloud_retriever_chunk_read_returns_full_manifest_text():
    retriever = _retriever()

    chunks = retriever.chunk_read(["c1", "missing"])

    assert chunks[0].text == "RAG 和 Qdrant 语义检索完整文本"


def test_cloud_retriever_retrieve_records_trace_and_metrics():
    retriever = _retriever()

    result = retriever.retrieve("RAG BM25")

    steps = [step.step for step in result.trace]
    assert "semantic_search" in steps
    assert "keyword_search" in steps
    assert "hybrid_fusion" in steps
    assert "rerank" in steps
    assert "chunk_read" in steps
    assert result.metrics["search_result_count"] >= 1
    assert result.metrics["read_chunk_count"] >= 1
