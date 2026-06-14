import json

from pydantic import SecretStr

from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.freshness.updater import CloudFreshnessUpdater
from kairorag.config import KairoCloudSettings


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def update_payload(self, chunk_ids, payload_patch, *, allowed_fields):
        self.calls.append((chunk_ids, payload_patch, allowed_fields))


def _settings(tmp_path):
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        qdrant_metadata_write_enabled=True,
        freshness_audit_log_path=str(tmp_path / "audit.jsonl"),
        observability_enabled=False,
    )


def _result():
    return JobFreshnessResult(
        job_id="job-1",
        chunk_id="chunk-1",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.example/old",
        status="closed",
        confidence=0.8,
        evidence=[],
        recommended_action="mark_closed",
        reason="测试关闭",
        checked_at="2026-06-12T00:00:00+00:00",
    )


def test_freshness_writeback_requires_confirmation_and_audits_decision(tmp_path):
    store = FakeVectorStore()
    updater = CloudFreshnessUpdater(_settings(tmp_path))

    result = updater.apply_update(_result(), vector_store=store, dry_run=False)
    lines = [json.loads(line) for line in open(result.audit_log_path, encoding="utf-8")]

    assert result.applied is False
    assert result.plan.dry_run is True
    assert store.calls == []
    assert lines[0]["approval_decision"]["requires_confirmation"] is True


def test_freshness_writeback_with_yes_applies(tmp_path):
    store = FakeVectorStore()
    updater = CloudFreshnessUpdater(_settings(tmp_path))

    result = updater.apply_update(_result(), vector_store=store, dry_run=False, user_confirmed=True)

    assert result.applied is True
    assert store.calls
    assert result.approval_decision.dry_run is False
