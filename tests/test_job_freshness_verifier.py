from pydantic import SecretStr

from kairorag.cloud.freshness.verifier import CloudJobFreshnessVerifier
from kairorag.config import KairoCloudSettings
from kairorag.providers.websearch import WebSearchResult


class FakeWebSearchProvider:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        return self.results


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _result(title, url, snippet, source="tavily"):
    return WebSearchResult(title=title, url=url, snippet=snippet, source=source)


def test_verifier_detects_active_signal_with_matching_job():
    provider = FakeWebSearchProvider(
        [_result("Kairo RAG Engineer", "https://jobs.kairo.ai/rag", "Job details. Apply now.")]
    )
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_by_query(
        "Kairo RAG Engineer",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.kairo.ai/rag",
    )

    assert result.status == "active"
    assert result.recommended_action == "keep_active"
    assert result.evidence[0].signal.startswith("active:")


def test_verifier_detects_closed_signal_before_active_signal():
    provider = FakeWebSearchProvider(
        [
            _result(
                "Kairo RAG Engineer",
                "https://jobs.kairo.ai/rag",
                "This job is no longer available. Job details are archived.",
            )
        ]
    )
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_by_query("Kairo RAG Engineer", company="Kairo", title="RAG Engineer")

    assert result.status == "closed"
    assert result.recommended_action == "mark_closed"


def test_verifier_detects_updated_url_with_active_signal():
    provider = FakeWebSearchProvider(
        [_result("Kairo RAG Engineer", "https://boards.example/new-rag", "Open position. Apply now.")]
    )
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_by_query(
        "Kairo RAG Engineer",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.kairo.ai/old-rag",
    )

    assert result.status == "updated"
    assert result.recommended_action == "update_url"
    assert result.new_url == "https://boards.example/new-rag"


def test_verifier_duplicate_metadata_skips_web_search():
    provider = FakeWebSearchProvider([])
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_from_metadata({"company": "Kairo", "title": "RAG Engineer", "duplicate_of": "job-0"})

    assert result.status == "duplicate"
    assert result.recommended_action == "mark_duplicate"
    assert provider.calls == []


def test_verifier_company_only_match_is_stale_not_active():
    provider = FakeWebSearchProvider([_result("Kairo careers", "https://kairo.ai/careers", "Kairo company page.")])
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_by_query("Kairo", company="Kairo", title="RAG Engineer")

    assert result.status == "stale"
    assert result.recommended_action == "mark_stale"


def test_verifier_no_matching_evidence_is_unknown():
    provider = FakeWebSearchProvider([_result("Other job", "https://example.com/job", "Apply now.")])
    verifier = CloudJobFreshnessVerifier(_settings(), provider)

    result = verifier.verify_by_query("Kairo RAG Engineer", company="Kairo", title="RAG Engineer")

    assert result.status == "unknown"
    assert result.recommended_action == "manual_review"
