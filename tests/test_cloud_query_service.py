import pytest
from pydantic import SecretStr

from kairorag.cloud.answering import CloudAnswer, CloudCitation
from kairorag.cloud.query import CloudQueryService
from kairorag.cloud.retriever import RetrievalResult
from kairorag.config import KairoCloudSettings


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, filters=None):
        self.calls.append((query, filters))
        return RetrievalResult(
            query=query,
            search_results=[],
            read_chunks=[],
            trace=[],
            metrics={"search_result_count": 1},
        )


class FakeAnswerGenerator:
    def answer(self, query, retrieval_result):
        del query, retrieval_result
        return CloudAnswer(
            answer="结构化回答",
            citations=[
                CloudCitation(
                    chunk_id="c1",
                    doc_id="d1",
                    title="文档",
                    evidence="证据",
                    metadata={},
                )
            ],
            retrieval_trace=[],
            metrics={"citation_count": 1},
        )


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_cloud_query_service_returns_structured_result():
    retriever = FakeRetriever()
    service = CloudQueryService(_settings(), retriever, FakeAnswerGenerator())

    result = service.ask("问题", filters={"company": "Kairo"})

    assert result.answer == "结构化回答"
    assert result.citations[0].chunk_id == "c1"
    assert result.metrics["retrieval_search_result_count"] == 1
    assert result.metrics["answer_citation_count"] == 1
    assert retriever.calls[0][1] == {"company": "Kairo"}


def test_cloud_query_service_rejects_blank_query():
    service = CloudQueryService(_settings(), FakeRetriever(), FakeAnswerGenerator())

    with pytest.raises(ValueError):
        service.ask("  ")
