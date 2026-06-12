import inspect

from kairorag.retrieval import keyword_search, semantic_search


def test_legacy_retrieval_modules_are_marked_deprecated():
    keyword_source = inspect.getsource(keyword_search)
    semantic_source = inspect.getsource(semantic_search)

    assert "Legacy / Deprecated" in keyword_source
    assert "Legacy / Deprecated" in semantic_source
