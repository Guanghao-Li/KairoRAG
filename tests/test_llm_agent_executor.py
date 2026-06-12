import pytest
from pydantic import SecretStr

from kairorag.cloud.agent.executor import LLMAgentExecutor
from kairorag.cloud.agent.planner import LLMToolPlanner
from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.answering import CloudAnswer, CloudCitation, SAFE_REFUSAL
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


class FakePlanner:
    def __init__(self, rounds):
        self.rounds = list(rounds)

    def next_tool_calls(self, state):
        if not self.rounds:
            return [LLMToolCall("finish", "finish", {"reason": "无更多工具"})]
        return self.rounds.pop(0)


class FakeRetriever:
    def hybrid_search(self, query, *, top_k, filters=None):
        return [_search_result("c1")]

    def semantic_search(self, query, *, top_k, filters=None):
        return [_search_result("c1")]

    def keyword_search(self, query, *, top_k, filters=None):
        return [_search_result("c1")]

    def chunk_read(self, chunk_ids):
        return [
            CloudReadChunk(
                chunk_id=chunk_id,
                doc_id="d1",
                title="岗位文档",
                text="岗位要求 RAG 和 LangGraph。完整证据文本。",
                metadata={"source_type": "job"},
            )
            for chunk_id in chunk_ids
        ]


class FakeAnswerGenerator:
    def __init__(self, answer="岗位要求 RAG。"):
        self.answer_text = answer

    def answer(self, query, retrieval_result):
        chunk = retrieval_result.read_chunks[0]
        return CloudAnswer(
            answer=self.answer_text,
            citations=[
                CloudCitation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    evidence="岗位要求 RAG",
                    metadata=chunk.metadata,
                )
            ],
            retrieval_trace=retrieval_result.trace,
            metrics={"citation_count": 1, "invalid_citation_count": 0, "llm_model": "fake"},
        )


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "max_tool_calls": 6,
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _search_result(chunk_id):
    return CloudSearchResult(
        chunk_id=chunk_id,
        doc_id="d1",
        title="岗位文档",
        text="岗位候选摘要",
        score=0.9,
        source="hybrid",
        metadata={"source_type": "job"},
    )


def _executor(rounds, *, settings=None, answer_generator=None):
    settings = settings or _settings()
    registry = ToolRegistry(settings, FakeRetriever(), answer_generator or FakeAnswerGenerator())
    return LLMAgentExecutor(settings, FakePlanner(rounds), registry)


def test_executor_normal_path_hybrid_read_generate():
    executor = _executor(
        [
            [LLMToolCall("1", "hybrid_search", {"query": "RAG"})],
            [LLMToolCall("2", "chunk_read", {"chunk_ids": ["c1"]})],
            [LLMToolCall("3", "generate_grounded_answer", {"query": "RAG"})],
        ]
    )

    result = executor.run("RAG")

    assert result.answer == "岗位要求 RAG。"
    assert result.citations[0].chunk_id == "c1"
    assert result.metrics["tool_call_count"] == 3


def test_executor_blocks_direct_generate_then_continues():
    executor = _executor(
        [
            [LLMToolCall("1", "generate_grounded_answer", {"query": "RAG"})],
            [LLMToolCall("2", "hybrid_search", {"query": "RAG"})],
            [LLMToolCall("3", "chunk_read", {"chunk_ids": ["c1"]})],
            [LLMToolCall("4", "generate_grounded_answer", {"query": "RAG"})],
        ]
    )

    result = executor.run("RAG")

    assert result.answer == "岗位要求 RAG。"
    assert result.metrics["blocked_final_answer_count"] == 1


def test_executor_rejects_unsearched_chunk_read():
    executor = _executor(
        [
            [LLMToolCall("1", "chunk_read", {"chunk_ids": ["missing"]})],
            [LLMToolCall("2", "finish", {"reason": "停止"})],
        ]
    )

    result = executor.run("RAG")

    assert result.answer == SAFE_REFUSAL
    assert result.metrics["invalid_tool_call_count"] == 1


def test_executor_rejects_unknown_tool():
    executor = _executor(
        [
            [LLMToolCall("1", "missing_tool", {})],
            [LLMToolCall("2", "finish", {"reason": "停止"})],
        ]
    )

    result = executor.run("RAG")

    assert result.metrics["invalid_tool_call_count"] == 1
    assert result.answer == SAFE_REFUSAL


def test_executor_stops_at_max_tool_calls_with_safe_refusal():
    executor = _executor(
        [[LLMToolCall("1", "hybrid_search", {"query": "RAG"})]],
        settings=_settings(max_tool_calls=1),
    )

    result = executor.run("RAG")

    assert result.answer == SAFE_REFUSAL
    assert result.metrics["max_tool_calls_reached"] is True


def test_executor_rejects_blank_query():
    executor = _executor([])

    with pytest.raises(ValueError):
        executor.run("  ")


def test_executor_freshness_query_does_not_claim_active_status():
    executor = _executor(
        [
            [LLMToolCall("1", "hybrid_search", {"query": "这个岗位还在招吗？"})],
            [LLMToolCall("2", "chunk_read", {"chunk_ids": ["c1"]})],
            [LLMToolCall("3", "generate_grounded_answer", {"query": "这个岗位还在招吗？"})],
        ],
        answer_generator=FakeAnswerGenerator(answer="这个岗位仍在招聘。"),
    )

    result = executor.run("这个岗位还在招吗？")

    assert "不能确认实时状态" in result.answer
    assert "仍在招聘" not in result.answer
    assert result.metrics["freshness_guardrail_applied"] is True
