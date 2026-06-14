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


def _settings(tmp_path, **overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "qdrant_metadata_write_enabled": False,
        "freshness_audit_log_path": str(tmp_path / "audit" / "freshness.jsonl"),
        "observability_enabled": False,
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _result(status, action, **overrides):
    values = {
        "job_id": "job-1",
        "chunk_id": "chunk-1",
        "company": "Kairo",
        "title": "RAG Engineer",
        "original_url": "https://jobs.example/old",
        "status": status,
        "confidence": 0.8,
        "evidence": [],
        "recommended_action": action,
        "reason": "测试原因 sk-secret-value。",
        "checked_at": "2026-06-12T00:00:00+00:00",
    }
    values.update(overrides)
    return JobFreshnessResult(**values)


def test_freshness_update_dry_run_does_not_write_qdrant(tmp_path):
    store = FakeVectorStore()
    updater = CloudFreshnessUpdater(_settings(tmp_path))

    result = updater.apply_update(_result("closed", "mark_closed"), vector_store=store)

    assert result.applied is False
    assert store.calls == []
    assert result.audit_log_path


def test_freshness_update_apply_writes_qdrant_when_enabled_and_confirmed(tmp_path):
    store = FakeVectorStore()
    updater = CloudFreshnessUpdater(_settings(tmp_path, qdrant_metadata_write_enabled=True))

    result = updater.apply_update(
        _result("closed", "mark_closed"),
        vector_store=store,
        dry_run=False,
        user_confirmed=True,
    )

    assert result.applied is True
    assert store.calls[0][0] == ["chunk-1"]
    assert store.calls[0][1]["verification_status"] == "closed"
    assert store.calls[0][1]["archived"] is True


def test_freshness_update_write_disabled_returns_error_after_confirmation(tmp_path):
    store = FakeVectorStore()
    updater = CloudFreshnessUpdater(_settings(tmp_path, qdrant_metadata_write_enabled=False))

    result = updater.apply_update(
        _result("closed", "mark_closed"),
        vector_store=store,
        dry_run=False,
        user_confirmed=True,
    )

    assert result.applied is False
    assert "未启用" in result.error
    assert store.calls == []


def test_freshness_update_audit_jsonl_redacts_secret_and_records_approval(tmp_path):
    updater = CloudFreshnessUpdater(_settings(tmp_path))

    result = updater.apply_update(_result("active", "keep_active"), vector_store=FakeVectorStore())
    lines = [json.loads(line) for line in open(result.audit_log_path, encoding="utf-8")]

    assert lines
    assert "sk-secret-value" not in json.dumps(lines, ensure_ascii=False)
    assert lines[0]["action"] == "keep_active"
    assert "approval_decision" in lines[0]


def test_freshness_update_status_patches(tmp_path):
    updater = CloudFreshnessUpdater(_settings(tmp_path))

    cases = [
        ("active", "keep_active", {"archived": False, "manual_review_required": False}),
        ("closed", "mark_closed", {"archived": True, "closed_reason": "测试原因 sk-secret-value。"}),
        ("updated", "update_url", {"original_url": "https://jobs.example/new", "canonical_url": "https://jobs.example/new"}),
        ("duplicate", "mark_duplicate", {"archived": True, "duplicate_of": "job-old"}),
        ("stale", "mark_stale", {"manual_review_required": True}),
        ("unknown", "manual_review", {"manual_review_required": True}),
    ]
    for status, action, expected in cases:
        extra = {}
        if status == "updated":
            extra["new_url"] = "https://jobs.example/new"
        if status == "duplicate":
            extra["new_url"] = "job-old"
        plan = updater.plan_update(_result(status, action, **extra))
        for key, value in expected.items():
            assert plan.metadata_patch[key] == value
