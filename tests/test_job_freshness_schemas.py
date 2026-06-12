import json

import pytest

from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult


def _evidence():
    return JobFreshnessEvidence(
        source="tavily",
        url="https://jobs.example/rag",
        title="RAG Engineer",
        snippet="Apply now",
        signal="active:apply now",
        confidence=0.8,
        observed_at="2026-06-12T00:00:00+00:00",
    )


def test_job_freshness_result_is_json_serializable():
    result = JobFreshnessResult(
        job_id="job-1",
        chunk_id="chunk-1",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.example/rag",
        status="active",
        confidence=0.8,
        evidence=[_evidence()],
        recommended_action="keep_active",
        reason="有明确申请信号。",
        checked_at="2026-06-12T00:00:00+00:00",
    )

    payload = result.to_dict()

    assert payload["status"] == "active"
    assert payload["evidence"][0]["signal"] == "active:apply now"
    assert json.loads(json.dumps(payload, ensure_ascii=False))["company"] == "Kairo"


def test_job_freshness_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        JobFreshnessEvidence(
            source="tavily",
            url="https://jobs.example/rag",
            title="RAG Engineer",
            snippet="Apply now",
            signal="active",
            confidence=1.1,
            observed_at="2026-06-12T00:00:00+00:00",
        )


def test_job_freshness_rejects_invalid_status_and_action():
    with pytest.raises(ValueError):
        JobFreshnessResult(
            job_id=None,
            chunk_id=None,
            company=None,
            title=None,
            original_url=None,
            status="paused",
            confidence=0.1,
            evidence=[],
            recommended_action="manual_review",
            reason="非法状态。",
            checked_at="2026-06-12T00:00:00+00:00",
        )
    with pytest.raises(ValueError):
        JobFreshnessResult(
            job_id=None,
            chunk_id=None,
            company=None,
            title=None,
            original_url=None,
            status="unknown",
            confidence=0.1,
            evidence=[],
            recommended_action="delete",
            reason="非法动作。",
            checked_at="2026-06-12T00:00:00+00:00",
        )
