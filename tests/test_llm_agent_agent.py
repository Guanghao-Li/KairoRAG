from pydantic import SecretStr

from kairorag.cloud.agent import CloudLLMAgent
from kairorag.cloud.answering import CloudAnswer, CloudCitation
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMResponse, LLMToolCall


class FakeLLMProvider:
    def __init__(self):
        self.responses = [
            LLMResponse("", "fake", {}, tool_calls=[LLMToolCall("1", "hybrid_search", {"query": "RAG"})]),
            LLMResponse("", "fake", {}, tool_calls=[LLMToolCall("2", "chunk_read", {"chunk_ids": ["c1"]})]),
            LLMResponse("", "fake", {}, tool_calls=[LLMToolCall("3", "generate_grounded_answer", {"query": "RAG"})]),
        ]

    def complete(self, messages, *, temperature=0.0, response_format=None):
        raise AssertionError("CloudLLMAgent planner 不应调用 complete")

    def complete_with_tools(self, messages, tools, *, temperature=0.0):
        return self.responses.pop(0)


class FakeRetriever:
    def hybrid_search(self, query, *, top_k, filters=None):
        return [CloudSearchResult("c1", "d1", "标题", "候选摘要", 0.9, "hybrid", {"source_type": "job"})]

    def semantic_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def keyword_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def chunk_read(self, chunk_ids):
        return [CloudReadChunk("c1", "d1", "标题", "完整证据文本，岗位要求 RAG。", {"source_type": "job"})]


class FakeAnswerGenerator:
    def answer(self, query, retrieval_result):
        return CloudAnswer(
            "岗位要求 RAG。",
            [CloudCitation("c1", "d1", "标题", "岗位要求 RAG", {"source_type": "job"})],
            retrieval_result.trace,
            {"citation_count": 1, "invalid_citation_count": 0, "llm_model": "fake"},
        )


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        max_tool_calls=6,
    )


def test_cloud_llm_agent_ask_returns_agent_run_result():
    agent = CloudLLMAgent(_settings(), FakeRetriever(), FakeAnswerGenerator(), FakeLLMProvider())

    result = agent.ask("RAG", filters={"source_type": "job"})

    assert result.answer == "岗位要求 RAG。"
    assert result.citations[0].chunk_id == "c1"
    assert result.metrics["tool_call_count"] == 3
    assert result.metrics["final_citation_count"] == 1
    assert any(step.step == "llm_planning" for step in result.trace)
