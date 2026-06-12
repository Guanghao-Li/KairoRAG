from pydantic import SecretStr

from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.answering import CloudAnswer
from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall
from kairorag.providers.websearch import WebSearchResult


class FakeRetriever:
    def hybrid_search(self, query, *, top_k, filters=None):
        return [
            CloudSearchResult("c1", "d1", "RAG 岗位", "候选摘要", 0.9, "hybrid", {"company": "Kairo"})
        ]

    def semantic_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def keyword_search(self, query, *, top_k, filters=None):
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    def chunk_read(self, chunk_ids):
        return [
            CloudReadChunk(
                chunk_id=chunk_id,
                doc_id="d1",
                title="RAG 岗位",
                text="Kairo RAG Engineer 岗位描述。",
                metadata={"company": "Kairo", "title": "RAG Engineer", "original_url": "https://jobs.kairo.ai/rag"},
            )
            for chunk_id in chunk_ids
        ]


class FakeAnswerGenerator:
    def answer(self, query, retrieval_result, *, freshness_results=None):
        return CloudAnswer("回答", [], retrieval_result.trace, {"ok": True}, list((freshness_results or {}).values()))


class FakeWebProvider:
    def __init__(self):
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        return [WebSearchResult("RAG", "https://jobs.example/rag", "Apply now", "fake")]


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_from_chunk(self, chunk):
        self.calls.append(("chunk", chunk.chunk_id))
        return _freshness_result(chunk_id=chunk.chunk_id)

    def verify_by_query(self, query, **kwargs):
        self.calls.append(("query", query, kwargs))
        return _freshness_result(chunk_id=None)


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _freshness_result(chunk_id):
    return JobFreshnessResult(
        job_id="job-1",
        chunk_id=chunk_id,
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.kairo.ai/rag",
        status="active",
        confidence=0.8,
        evidence=[
            JobFreshnessEvidence(
                "fake",
                "https://jobs.kairo.ai/rag",
                "RAG Engineer",
                "Apply now",
                "active:apply now",
                0.8,
                "2026-06-12T00:00:00+00:00",
            )
        ],
        recommended_action="keep_active",
        reason="有申请信号。",
        checked_at="2026-06-12T00:00:00+00:00",
    )


def _registry(settings=None, web_provider=None, verifier=None):
    return ToolRegistry(
        settings or _settings(),
        FakeRetriever(),
        FakeAnswerGenerator(),
        web_search_provider=web_provider,
        freshness_verifier=verifier,
    )


def test_web_search_tool_calls_provider_without_raw():
    provider = FakeWebProvider()
    registry = _registry(web_provider=provider, verifier=FakeVerifier())
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "web_search", {"query": "RAG", "max_results": 2}), state)

    assert output.ok is True
    assert provider.calls == [("RAG", 2)]
    assert "raw" not in output.result["results"][0]


def test_verify_job_freshness_saves_state_result_from_read_chunk():
    verifier = FakeVerifier()
    registry = _registry(web_provider=FakeWebProvider(), verifier=verifier)
    state = AgentState(query="这个岗位还在招吗？")
    registry.execute(LLMToolCall("1", "hybrid_search", {"query": "RAG"}), state)
    registry.execute(LLMToolCall("2", "chunk_read", {"chunk_ids": ["c1"]}), state)

    output = registry.execute(LLMToolCall("3", "verify_job_freshness", {"chunk_id": "c1"}), state)

    assert output.ok is True
    assert verifier.calls == [("chunk", "c1")]
    assert state.freshness_results["c1"].status == "active"


def test_verify_job_freshness_by_query_calls_verifier():
    verifier = FakeVerifier()
    registry = _registry(web_provider=FakeWebProvider(), verifier=verifier)
    state = AgentState(query="Kairo RAG Engineer 还开放吗？")

    output = registry.execute(
        LLMToolCall("1", "verify_job_freshness", {"company": "Kairo", "title": "RAG Engineer"}),
        state,
    )

    assert output.ok is True
    assert verifier.calls[0][0] == "query"
    assert state.metrics["freshness_verification_count"] == 1


def test_freshness_tools_error_when_disabled():
    registry = _registry(settings=_settings(job_freshness_enabled=False), web_provider=None, verifier=None)
    state = AgentState(query="RAG")

    output = registry.execute(LLMToolCall("1", "verify_job_freshness", {"query": "RAG"}), state)

    assert output.ok is False
    assert "未启用" in output.error


def test_verify_job_freshness_requires_read_chunk_for_chunk_id():
    registry = _registry(web_provider=FakeWebProvider(), verifier=FakeVerifier())
    state = AgentState(query="这个岗位还在招吗？")
    state.search_results["c1"] = CloudSearchResult("c1", "d1", "RAG", "摘要", 0.9, "hybrid", {})

    output = registry.execute(LLMToolCall("1", "verify_job_freshness", {"chunk_id": "c1"}), state)

    assert output.ok is False
    assert "chunk_read" in output.error
