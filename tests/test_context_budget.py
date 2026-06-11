from kairorag.retrieval.context_budget import ContextBudgetManager
from kairorag.schemas import ReadChunkResult, SearchResult


def test_context_budget_limits_and_deduplicates():
    manager = ContextBudgetManager(max_search_results=2, max_chunks_to_read=1, max_context_tokens=20)
    results = [
        SearchResult("c1", 1.0, "job", "A", "one"),
        SearchResult("c1", 0.9, "job", "A", "duplicate"),
        SearchResult("c2", 0.8, "job", "B", "two"),
        SearchResult("c3", 0.7, "job", "C", "three"),
    ]
    limited = manager.limit_search_results(results)
    assert [result.chunk_id for result in limited] == ["c1", "c2"]
    assert manager.can_read("c1")
    assert manager.record_read(ReadChunkResult("c1", "short text"))
    assert not manager.can_read("c2")

