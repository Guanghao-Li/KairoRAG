from types import SimpleNamespace

from pydantic import SecretStr

from kairorag.cloud import query_cli
from kairorag.cloud.agent import cli as agent_cli
from kairorag.cloud.agent.schemas import AgentRunResult
from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult
from kairorag.config import KairoCloudSettings


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def _verification_result():
    return JobFreshnessResult(
        job_id="job-1",
        chunk_id="c1",
        company="Kairo",
        title="RAG Engineer",
        original_url="https://jobs.example/rag",
        status="active",
        confidence=0.8,
        evidence=[
            JobFreshnessEvidence(
                "fake",
                "https://jobs.example/rag",
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


class FakeAgent:
    def __init__(self):
        self.calls = []

    def ask(self, query, *, filters=None):
        self.calls.append((query, filters))
        return AgentRunResult(
            query=query,
            answer="Agent 回答",
            citations=[],
            trace=[],
            metrics={"ok": True},
            verification_results=[_verification_result()],
        )


def test_agent_cli_verify_freshness_and_show_verification(monkeypatch, capsys):
    settings = _settings()
    agent = FakeAgent()
    loaded_settings = []
    monkeypatch.setattr(agent_cli, "KairoCloudSettings", lambda: settings)
    monkeypatch.setattr(
        agent_cli,
        "build_cloud_agent_runtime",
        lambda loaded: loaded_settings.append(loaded) or SimpleNamespace(settings=settings, llm_agent=agent),
    )

    assert agent_cli.main(["这个岗位还在招吗？", "--verify-freshness", "--show-verification", "--web-provider", "bing"]) == 0
    output = capsys.readouterr().out

    assert "Verification" in output
    assert "status=active" in output
    assert loaded_settings[0].web_search_provider == "bing"


def test_query_cli_show_verification_in_default_agent_mode(monkeypatch, capsys):
    settings = _settings()
    agent = FakeAgent()
    monkeypatch.setattr(query_cli, "KairoCloudSettings", lambda: settings)
    monkeypatch.setattr(
        query_cli,
        "build_cloud_agent_runtime",
        lambda loaded: SimpleNamespace(settings=settings, llm_agent=agent),
    )

    assert query_cli.main(["Kairo RAG Engineer 还开放吗？", "--verify-freshness", "--show-verification"]) == 0
    output = capsys.readouterr().out

    assert "Agent 回答" in output
    assert "status=active" in output


def test_agent_cli_verify_freshness_errors_when_disabled(monkeypatch, capsys):
    monkeypatch.setattr(agent_cli, "KairoCloudSettings", lambda: _settings(job_freshness_enabled=False))

    assert agent_cli.main(["这个岗位还在招吗？", "--verify-freshness"]) == 2
    output = capsys.readouterr().out

    assert "JOB_FRESHNESS_ENABLED=false" in output


def test_query_cli_rejects_baseline_with_verify_freshness(monkeypatch, capsys):
    monkeypatch.setattr(query_cli, "KairoCloudSettings", lambda: _settings())

    assert query_cli.main(["这个岗位还在招吗？", "--baseline", "--verify-freshness"]) == 2
    output = capsys.readouterr().out

    assert "baseline" in output
