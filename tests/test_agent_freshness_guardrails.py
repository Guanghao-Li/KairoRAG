from pydantic import SecretStr

from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.answering import CloudAnswer
from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


class FakeRetriever:
    def hybrid_search(self, query, *, top_k, filters=None):
        return [CloudSearchResult("c1", "d1", "RAG 岗位", "摘要", 0.9, "hybrid", {})]

    def semantic_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def keyword_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def chunk_read(self, chunk_ids):
        return [CloudReadChunk(chunk_ids[0], "d1", "RAG 岗位", "Kairo RAG Engineer 岗位描述。", {})]


class ClaimingAnswerGenerator:
    def __init__(self, answer):
        self.answer_text = answer

    def answer(self, query, retrieval_result, *, freshness_results=None):
        return CloudAnswer(self.answer_text, [], retrieval_result.trace, {"citation_count": 0})


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _registry(answer):
    return ToolRegistry(_settings(), FakeRetriever(), ClaimingAnswerGenerator(answer))


def _state(query="这个岗位现在还在招吗？"):
    state = AgentState(query=query)
    state.search_results["c1"] = CloudSearchResult("c1", "d1", "RAG 岗位", "摘要", 0.9, "hybrid", {})
    state.read_chunks["c1"] = CloudReadChunk("c1", "d1", "RAG 岗位", "Kairo RAG Engineer 岗位描述。", {})
    return state


def _freshness(status, action, confidence=0.8):
    evidence = []
    if status == "active":
        evidence = [
            JobFreshnessEvidence(
                "fake",
                "https://jobs.example/rag",
                "RAG Engineer",
                "Apply now",
                "active:apply now",
                0.8,
                "2026-06-12T00:00:00+00:00",
            )
        ]
    return JobFreshnessResult(
        job_id="job-1",
        chunk_id="c1",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.example/rag",
        status=status,
        confidence=confidence,
        evidence=evidence,
        recommended_action=action,
        reason="测试。",
        checked_at="2026-06-12T00:00:00+00:00",
    )


def test_freshness_query_without_verification_blocks_final_answer():
    registry = _registry("这个岗位还在招。")
    state = _state()

    output = registry.execute(LLMToolCall("1", "generate_grounded_answer", {"query": state.query}), state)

    assert output.ok is False
    assert "verify_job_freshness" in output.error


def test_closed_verification_prevents_apply_recommendation():
    registry = _registry("这个岗位还在招，可以申请。")
    state = _state()
    state.freshness_results["c1"] = _freshness("closed", "mark_closed")

    output = registry.execute(LLMToolCall("1", "generate_grounded_answer", {"query": state.query}), state)

    assert output.ok is True
    assert "已关闭" in state.final_answer
    assert "还在招" not in state.final_answer


def test_stale_verification_requires_uncertainty_statement():
    registry = _registry("基于岗位描述可以继续了解。")
    state = _state()
    state.freshness_results["c1"] = _freshness("stale", "mark_stale", confidence=0.2)

    registry.execute(LLMToolCall("1", "generate_grounded_answer", {"query": state.query}), state)

    assert "不能确认岗位实时状态" in state.final_answer


def test_active_verification_includes_evidence_framing():
    registry = _registry("岗位信息匹配。")
    state = _state()
    state.freshness_results["c1"] = _freshness("active", "keep_active")

    registry.execute(LLMToolCall("1", "generate_grounded_answer", {"query": state.query}), state)

    assert "web verification evidence" in state.final_answer
