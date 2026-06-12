from pydantic import SecretStr

from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.freshness.updater import CloudFreshnessUpdater
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


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
        "qdrant_metadata_write_enabled": True,
        "freshness_audit_log_path": str(tmp_path / "freshness_audit.jsonl"),
        "observability_enabled": False,
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _freshness_result(chunk_id="chunk-1", status="closed"):
    return JobFreshnessResult(
        job_id="job-1",
        chunk_id=chunk_id,
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.example/old",
        status=status,
        confidence=0.8,
        evidence=[],
        recommended_action="mark_closed" if status == "closed" else "keep_active",
        reason="测试。",
        checked_at="2026-06-12T00:00:00+00:00",
    )


def _registry(tmp_path, *, authorized=False, settings=None, store=None):
    settings = settings or _settings(tmp_path)
    return ToolRegistry(
        settings,
        retriever=object(),
        answer_generator=object(),
        freshness_updater=CloudFreshnessUpdater(settings),
        vector_store=store or FakeVectorStore(),
        freshness_apply_authorized=authorized,
    )


def test_apply_freshness_update_dry_run_plan_success(tmp_path):
    store = FakeVectorStore()
    registry = _registry(tmp_path, store=store)
    state = AgentState(query="检查岗位")
    state.freshness_results["chunk-1"] = _freshness_result()

    output = registry.execute(LLMToolCall("1", "apply_freshness_update", {"chunk_id": "chunk-1"}), state)

    assert output.ok is True
    assert output.result["applied"] is False
    assert store.calls == []
    assert output.result["audit_log_path"]


def test_apply_freshness_update_fails_without_freshness_result(tmp_path):
    registry = _registry(tmp_path)
    state = AgentState(query="检查岗位")

    output = registry.execute(LLMToolCall("1", "apply_freshness_update", {"chunk_id": "chunk-1"}), state)

    assert output.ok is False
    assert "尚无 freshness" in output.error


def test_apply_freshness_update_unauthorized_apply_forces_dry_run(tmp_path):
    store = FakeVectorStore()
    registry = _registry(tmp_path, authorized=False, store=store)
    state = AgentState(query="检查岗位")
    state.freshness_results["chunk-1"] = _freshness_result()

    output = registry.execute(
        LLMToolCall("1", "apply_freshness_update", {"chunk_id": "chunk-1", "dry_run": False}),
        state,
    )

    assert output.ok is True
    assert output.result["effective_dry_run"] is True
    assert store.calls == []
    assert "guardrail" in output.result


def test_apply_freshness_update_authorized_apply_writes(tmp_path):
    store = FakeVectorStore()
    registry = _registry(tmp_path, authorized=True, store=store)
    state = AgentState(query="请写回并标记关闭")
    state.freshness_results["chunk-1"] = _freshness_result()

    output = registry.execute(
        LLMToolCall("1", "apply_freshness_update", {"chunk_id": "chunk-1", "dry_run": False}),
        state,
    )

    assert output.ok is True
    assert output.result["applied"] is True
    assert store.calls


def test_apply_freshness_update_write_disabled_returns_error(tmp_path):
    store = FakeVectorStore()
    settings = _settings(tmp_path, qdrant_metadata_write_enabled=False)
    registry = _registry(tmp_path, authorized=True, settings=settings, store=store)
    state = AgentState(query="请写回")
    state.freshness_results["chunk-1"] = _freshness_result()

    output = registry.execute(
        LLMToolCall("1", "apply_freshness_update", {"chunk_id": "chunk-1", "dry_run": False}),
        state,
    )

    assert output.ok is False
    assert "未启用" in output.error
    assert store.calls == []
