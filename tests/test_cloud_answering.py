from pydantic import SecretStr

from kairorag.cloud.answering import SAFE_REFUSAL, GroundedCloudAnswerGenerator
from kairorag.cloud.retriever import CloudReadChunk, RetrievalResult
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMResponse


class FakeLLMProvider:
    model = "fake-llm"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, messages, *, temperature=0.0, response_format=None):
        self.calls.append((messages, temperature, response_format))
        return LLMResponse(content=self.content, model=self.model, usage={})


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def _retrieval(read_chunks):
    return RetrievalResult(
        query="哪些岗位要求 RAG？",
        search_results=[],
        read_chunks=read_chunks,
        trace=[],
        metrics={},
    )


def _chunk():
    return CloudReadChunk(
        chunk_id="c1",
        doc_id="d1",
        title="岗位文档",
        text="岗位要求 RAG、LangGraph 和 Qdrant 经验。",
        metadata={"company": "Kairo"},
    )


def test_answering_refuses_without_read_chunks_and_skips_llm():
    llm = FakeLLMProvider('{"answer":"不会被调用","citations":[]}')
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("问题", _retrieval([]))

    assert answer.answer == SAFE_REFUSAL
    assert llm.calls == []


def test_answering_accepts_valid_citation_from_read_chunk():
    llm = FakeLLMProvider(
        '{"answer":"岗位要求 RAG。","citations":[{"chunk_id":"c1","evidence":"岗位要求 RAG"}]}'
    )
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("问题", _retrieval([_chunk()]))

    assert answer.answer == "岗位要求 RAG。"
    assert answer.citations[0].chunk_id == "c1"
    assert answer.metrics["invalid_citation_count"] == 0


def test_answering_filters_missing_chunk_citation():
    llm = FakeLLMProvider(
        '{"answer":"岗位要求 RAG。","citations":[{"chunk_id":"missing","evidence":"岗位要求 RAG"}]}'
    )
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("问题", _retrieval([_chunk()]))

    assert answer.citations == []
    assert answer.metrics["invalid_citation_count"] == 1


def test_answering_safe_fails_on_invalid_json():
    llm = FakeLLMProvider("不是 JSON")
    generator = GroundedCloudAnswerGenerator(_settings(), llm)

    answer = generator.answer("问题", _retrieval([_chunk()]))

    assert answer.answer == SAFE_REFUSAL
    assert answer.citations == []
    assert answer.metrics["invalid_json"] is True
