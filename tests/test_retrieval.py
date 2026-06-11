from kairorag.indexing.build_index import build_indexes
from kairorag.retrieval.chunk_read import chunk_read
from kairorag.retrieval.hybrid_search import hybrid_search
from kairorag.retrieval.keyword_search import keyword_search
from kairorag.retrieval.semantic_search import semantic_search


def test_retrieval_tools_exclude_archived_by_default(tmp_path):
    build_indexes("data/raw", tmp_path)
    keyword = keyword_search("VectorWorks", top_k=5, index_dir=tmp_path)
    assert all(result.metadata.get("verification_status") != "closed" for result in keyword)

    archived = keyword_search("VectorWorks", top_k=5, include_archived=True, index_dir=tmp_path)
    assert any(result.metadata.get("verification_status") == "closed" for result in archived)


def test_semantic_hybrid_and_chunk_read(tmp_path):
    build_indexes("data/raw", tmp_path)
    semantic = semantic_search("agentic retrieval citations", top_k=5, index_dir=tmp_path)
    assert semantic
    hybrid = hybrid_search("RAG LangGraph multi agent", top_k=5, index_dir=tmp_path)
    assert len({result.chunk_id for result in hybrid}) == len(hybrid)
    chunk = chunk_read(hybrid[0].chunk_id, index_dir=tmp_path)
    assert chunk.text
    assert chunk.metadata["title"]

