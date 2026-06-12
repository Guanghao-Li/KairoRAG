from types import SimpleNamespace

from pydantic import SecretStr

from kairorag.cloud.agent import cli as agent_cli
from kairorag.cloud.agent.schemas import AgentRunResult
from kairorag.config import KairoCloudSettings


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_agent_cli_parses_arguments_and_prints_result(monkeypatch, capsys):
    settings = _settings()

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def ask(self, query, *, filters=None):
            self.calls.append((query, filters))
            return AgentRunResult(query=query, answer="Agent CLI 回答", citations=[], trace=[], metrics={"ok": True})

    agent = FakeAgent()
    monkeypatch.setattr(
        agent_cli,
        "build_cloud_agent_runtime",
        lambda loaded: SimpleNamespace(settings=settings, llm_agent=agent),
    )

    assert (
        agent_cli.main(
            [
                "哪些岗位要求 RAG？",
                "--top-k",
                "4",
                "--max-tool-calls",
                "5",
                "--source-type",
                "job",
                "--company",
                "Kairo",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Agent CLI 回答" in output
    assert agent.calls[0] == ("哪些岗位要求 RAG？", {"source_type": "job", "company": "Kairo"})
    assert settings.max_search_results == 4
    assert settings.max_tool_calls == 5
