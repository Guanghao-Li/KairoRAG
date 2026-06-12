import pytest
from pydantic import SecretStr

from kairorag.cloud.indexing import CloudIndexer
from kairorag.cloud.manifest import load_manifest
from kairorag.config import KairoCloudSettings
from kairorag.providers.embeddings import EmbeddingResult
from kairorag.providers.errors import KairoProviderError
from kairorag.schemas import DocumentRecord


class FakeEmbeddingProvider:
    model = "fake-embedding"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [
            EmbeddingResult(text=text, vector=[float(index + 1), 0.5], model=self.model, usage={})
            for index, text in enumerate(texts)
        ]

    def embed_query(self, query):
        return self.embed_texts([query])[0]


class FakeVectorStore:
    def __init__(self):
        self.vector_size = None
        self.upserted = []

    def ensure_collection(self, vector_size):
        self.vector_size = vector_size

    def upsert_chunks(self, chunks):
        self.upserted.extend(chunks)

    def search(self, query_vector, *, top_k, filters=None):
        return []

    def delete_chunks(self, chunk_ids):
        del chunk_ids

    def healthcheck(self):
        return True


class FakeKeywordSearch:
    def __init__(self):
        self.documents = []

    def index(self, documents):
        self.documents = list(documents)

    def search(self, query, *, top_k, filters=None):
        return []


def _settings(tmp_path):
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        cloud_index_manifest_path=str(tmp_path / "manifest.json"),
        cloud_bm25_index_path=str(tmp_path / "bm25.json"),
        cloud_chunk_batch_size=2,
    )


def test_cloud_indexer_builds_manifest_vector_and_bm25(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "kairorag.cloud.indexing.load_documents",
        lambda raw_dir: [
            DocumentRecord(
                doc_id="doc-1",
                source_type="company_doc",
                title="Kairo 文档",
                text="RAG 检索 需要 Qdrant 和 BM25 融合。",
                metadata={"company": "Kairo"},
            )
        ],
    )
    settings = _settings(tmp_path)
    embedding = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    keyword = FakeKeywordSearch()
    indexer = CloudIndexer(settings, embedding, vector_store, keyword)

    stats = indexer.build()

    assert stats.documents_loaded == 1
    assert stats.chunks_created >= 1
    assert stats.embeddings_created == stats.chunks_created
    assert vector_store.vector_size == 2
    assert len(vector_store.upserted) == stats.chunks_created
    assert len(keyword.documents) == stats.chunks_created
    manifest = load_manifest(settings.cloud_index_manifest_path)
    assert manifest.chunks[0].text
    assert (tmp_path / "bm25.json").exists()
    assert not (tmp_path / "keyword_index.pkl").exists()
    assert not (tmp_path / "vector_store.pkl").exists()


def test_cloud_indexer_rejects_empty_documents(monkeypatch, tmp_path):
    monkeypatch.setattr("kairorag.cloud.indexing.load_documents", lambda raw_dir: [])
    indexer = CloudIndexer(_settings(tmp_path), FakeEmbeddingProvider(), FakeVectorStore(), FakeKeywordSearch())

    with pytest.raises(KairoProviderError) as exc_info:
        indexer.build()

    assert "未加载到任何知识库文档" in str(exc_info.value)
