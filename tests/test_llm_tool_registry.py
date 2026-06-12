from pydantic import SecretStr

from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.answering import CloudAnswer, CloudCitation
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


class FakeRetriever:
    def semantic_search(self, query, *, top_k, filters=None):
        return [_search_result("c1", "semantic", query)]

    def keyword_search(self, query, *, top_k, filters=None):
        return [_search_result("c2", "keyword", query)]

    def hybrid_search(self, query, *, top_k, filters=None):
        return [_search_result("c1", "hybrid", query)]

    def chunk_read(self, chunk_ids):
        return [
            CloudReadChunk(
                chunk_id=chunk_id,
                doc_id=f"doc-{chunk_id}",
                title=f"标题 {chunk_id}",
                text=f"{chunk_id} 完整证据文本，包含 RAG。",
                metadata={"source_type": "job"},
            )
            for chunk_id in chunk_ids
        ]


class FakeAnswerGenerator:
    def answer(self, query, retrieval_result):
        chunk = retrieval_result.read_chunks[0]
        return CloudAnswer(
            answer=f"基于 {chunk.chunk_id} 回答。",
            citations=[
                CloudCitation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    evidence="完整证据文本",
                    metadata=chunk.metadata,
                )
            ],
            retrieval_trace=retrieval_result.trace,
            metrics={"citation_count": 1, "invalid_citation_count": 0, "llm_model": "fake"},
        )


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def _registry():
    return ToolRegistry(_settings(), FakeRetriever(), FakeAnswerGenerator())


def _search_result(chunk_id, source, query):
    return CloudSearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title=f"标题 {chunk_id}",
        text=f"{query} 的候选摘要文本",
        score=0.9,
        source=source,
        metadata={"source_type": "job"},
    )


def test_tool_registry_specs_return_all_tools():
    names = {spec.name for spec in _registry().specs()}

    assert names == {
        "semantic_search",
        "keyword_search",
        "hybrid_search",
        "chunk_read",
        "web_search",
        "verify_job_freshness",
        "generate_grounded_answer",
        "finish",
    }


def test_tool_registry_semantic_and_hybrid_save_search_results():
    registry = _registry()
    state = AgentState(query="RAG")

    registry.execute(LLMToolCall("1", "semantic_search", {"query": "RAG"}), state)
    registry.execute(LLMToolCall("2", "hybrid_search", {"query": "RAG"}), state)

    assert "c1" in state.search_results
    assert state.metrics["search_result_count"] == 1


def test_tool_registry_chunk_read_only_reads_searched_chunks():
    registry = _registry()
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "chunk_read", {"chunk_ids": ["missing"]}), state)

    assert output.ok is False
    assert state.invalid_tool_call_count == 1

    registry.execute(LLMToolCall("2", "hybrid_search", {"query": "RAG"}), state)
    output = registry.execute(LLMToolCall("3", "chunk_read", {"chunk_ids": ["c1"]}), state)

    assert output.ok is True
    assert "c1" in state.read_chunks


def test_tool_registry_generate_answer_requires_read_chunks():
    registry = _registry()
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "generate_grounded_answer", {"query": "RAG"}), state)

    assert output.ok is False
    assert state.blocked_final_answer_count == 1


def test_tool_registry_generate_answer_succeeds_with_read_chunks():
    registry = _registry()
    state = AgentState(query="RAG")
    registry.execute(LLMToolCall("1", "hybrid_search", {"query": "RAG"}), state)
    registry.execute(LLMToolCall("2", "chunk_read", {"chunk_ids": ["c1"]}), state)

    output = registry.execute(LLMToolCall("3", "generate_grounded_answer", {"query": "RAG"}), state)

    assert output.ok is True
    assert state.finished is True
    assert state.final_citations[0].chunk_id == "c1"


def test_tool_registry_unknown_tool_is_invalid():
    registry = _registry()
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "missing_tool", {}), state)

    assert output.ok is False
    assert state.invalid_tool_call_count == 1
