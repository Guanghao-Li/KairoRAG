from pydantic import SecretStr

from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.answering import CloudAnswer, SAFE_REFUSAL
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


class LongTextRetriever:
    def hybrid_search(self, query, *, top_k, filters=None):
        return [
            CloudSearchResult(
                chunk_id="c1",
                doc_id="d1",
                title="长文档",
                text="候选摘要" * 200,
                score=0.9,
                source="hybrid",
                metadata={},
            )
        ]

    def semantic_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def keyword_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def chunk_read(self, chunk_ids):
        return [CloudReadChunk(chunk_ids[0], "d1", "长文档", "完整文本" * 300, {})]


class NoopAnswerGenerator:
    def answer(self, query, retrieval_result):
        return CloudAnswer("回答", [], retrieval_result.trace, {"citation_count": 0})


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_finish_cannot_accept_planner_written_answer():
    registry = ToolRegistry(_settings(), LongTextRetriever(), NoopAnswerGenerator())
    state = AgentState(query="RAG")

    output = registry.execute(
        LLMToolCall("1", "finish", {"reason": "最终答案：这个答案来自 planner"}),
        state,
    )

    assert output.ok is True
    assert state.final_answer == SAFE_REFUSAL


def test_tool_result_limits_search_text_length():
    registry = ToolRegistry(_settings(), LongTextRetriever(), NoopAnswerGenerator())
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "hybrid_search", {"query": "RAG"}), state)

    preview = output.result["results"][0]["text_preview"]
    assert len(preview) < 260
    assert output.result["results"][0]["可作为最终证据"] is False


def test_trace_hides_sensitive_field_names():
    registry = ToolRegistry(_settings(), LongTextRetriever(), NoopAnswerGenerator())
    state = AgentState(query="RAG")

    registry.execute(LLMToolCall("1", "hybrid_search", {"query": "OPENAI_API_KEY=secret"}), state)

    assert any(step.detail.get("已隐藏") for step in state.trace)
