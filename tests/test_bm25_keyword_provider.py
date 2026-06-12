from kairorag.providers.keyword import BM25KeywordSearchProvider, KeywordDocument


def test_bm25_keyword_provider_finds_relevant_chunk():
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "OpenAI RAG", "agentic retrieval with citations", {"source_type": "note"}),
            KeywordDocument("c2", "d2", "Payroll", "invoice workflow", {"source_type": "note"}),
            KeywordDocument("c3", "d3", "Travel", "policy document", {"source_type": "note"}),
        ]
    )

    results = provider.search("agentic citations", top_k=2)

    assert results[0].chunk_id == "c1"


def test_bm25_keyword_provider_filters_archived_documents():
    provider = BM25KeywordSearchProvider()
    provider.index(
        [
            KeywordDocument("c1", "d1", "Legacy", "qdrant migration", {"archived": True, "source_type": "job"}),
            KeywordDocument("c2", "d2", "Cloud", "qdrant migration", {"archived": False, "source_type": "job"}),
            KeywordDocument("c3", "d3", "Other", "different text", {"archived": False, "source_type": "note"}),
        ]
    )

    results = provider.search("qdrant", top_k=5, filters={"archived": False})

    assert [result.chunk_id for result in results] == ["c2"]


def test_bm25_keyword_provider_boosts_title_matches():
    provider = BM25KeywordSearchProvider(title_weight=4)
    provider.index(
        [
            KeywordDocument("title-hit", "d1", "Qdrant tuning", "vector database", {"source_type": "note"}),
            KeywordDocument("body-hit", "d2", "Other topic", "qdrant", {"source_type": "note"}),
            KeywordDocument("miss", "d3", "Other", "unrelated", {"source_type": "note"}),
        ]
    )

    results = provider.search("qdrant", top_k=2)

    assert results[0].chunk_id == "title-hit"
