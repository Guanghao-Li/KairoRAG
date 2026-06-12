from pydantic import SecretStr

from kairorag.cloud.answering import GroundedCloudAnswerGenerator
from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk, RetrievalResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMResponse


class FakeLLMProvider:
    model = "fake-llm"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, messages, *, temperature=0.0, response_format=None):
        self.calls.append(messages)
        return LLMResponse(content=self.content, model=self.model, usage={})


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


def _chunk():
    return CloudReadChunk("c1", "d1", "RAG 岗位", "Kairo RAG Engineer 岗位描述。", {"company": "Kairo"})


def _retrieval():
    return RetrievalResult("岗位还开放吗？", [], [_chunk()], [], {})


def _freshness(status, action, confidence=0.8):
    evidence = [
        JobFreshnessEvidence(
            "fake",
            "https://jobs.example/rag",
            "RAG Engineer",
            "Apply now",
            f"{status}:signal",
            confidence,
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


def test_answering_closed_verification_blocks_active_claim():
    llm = FakeLLMProvider(
        '{"answer":"这个岗位当前可申请。","citations":[{"chunk_id":"c1","evidence":"Kairo RAG Engineer"}]}'
    )
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("岗位还开放吗？", _retrieval(), freshness_results={"c1": _freshness("closed", "mark_closed")})

    assert "已关闭" in answer.answer
    assert "当前可申请。" not in answer.answer


def test_answering_stale_unknown_mentions_unconfirmed_status():
    llm = FakeLLMProvider('{"answer":"可以参考岗位描述。","citations":[{"chunk_id":"c1","evidence":"Kairo RAG Engineer"}]}')
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("岗位还开放吗？", _retrieval(), freshness_results={"c1": _freshness("unknown", "manual_review", 0.0)})

    assert "不能确认岗位实时状态" in answer.answer


def test_answering_active_mentions_web_verification_evidence():
    llm = FakeLLMProvider('{"answer":"岗位描述匹配。","citations":[{"chunk_id":"c1","evidence":"Kairo RAG Engineer"}]}')
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("岗位还开放吗？", _retrieval(), freshness_results={"c1": _freshness("active", "keep_active")})

    assert "web verification evidence" in answer.answer


def test_answering_freshness_evidence_is_not_chunk_citation():
    llm = FakeLLMProvider(
        '{"answer":"岗位描述匹配。","citations":[{"chunk_id":"web-1","evidence":"Apply now"},{"chunk_id":"c1","evidence":"Kairo RAG Engineer"}]}'
    )
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("岗位还开放吗？", _retrieval(), freshness_results={"c1": _freshness("active", "keep_active")})

    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert answer.verification_results[0].evidence[0].url == "https://jobs.example/rag"
