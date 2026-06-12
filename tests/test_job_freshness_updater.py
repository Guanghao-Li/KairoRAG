from pydantic import SecretStr

from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.freshness.updater import CloudFreshnessUpdater
from kairorag.config import KairoCloudSettings


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


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
        "reason": "测试。",
        "checked_at": "2026-06-12T00:00:00+00:00",
    }
    values.update(overrides)
    return JobFreshnessResult(**values)


def test_updater_active_keeps_unarchived():
    plan = CloudFreshnessUpdater(_settings()).plan_update(_result("active", "keep_active"))

    assert plan["action"] == "keep_active"
    assert plan["metadata_patch"]["archived"] is False
    assert plan["requires_reindex"] is False


def test_updater_closed_archives_and_requires_reindex():
    plan = CloudFreshnessUpdater(_settings()).plan_update(_result("closed", "mark_closed"))

    assert plan["metadata_patch"]["archived"] is True
    assert plan["requires_reindex"] is True


def test_updater_updated_patches_new_url():
    plan = CloudFreshnessUpdater(_settings()).plan_update(
        _result("updated", "update_url", new_url="https://jobs.example/new")
    )

    assert plan["metadata_patch"]["original_url"] == "https://jobs.example/new"
    assert plan["requires_reindex"] is True


def test_updater_duplicate_archives():
    plan = CloudFreshnessUpdater(_settings()).plan_update(_result("duplicate", "mark_duplicate"))

    assert plan["metadata_patch"]["archived"] is True


def test_updater_stale_marks_review_without_reindex():
    plan = CloudFreshnessUpdater(_settings()).plan_update(_result("stale", "mark_stale", confidence=0.2))

    assert plan["action"] == "mark_stale"
    assert plan["requires_reindex"] is False
