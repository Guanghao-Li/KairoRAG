from types import SimpleNamespace

from pydantic import SecretStr

from kairorag.cloud import index as index_cli
from kairorag.cloud import query_cli
from kairorag.cloud.indexing import CloudIndexStats
from kairorag.cloud.query import CloudQueryResult
from kairorag.cloud.agent.schemas import AgentRunResult
from kairorag.config import KairoCloudSettings


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_cloud_index_cli_calls_indexer(monkeypatch, capsys):
    settings = _settings()
    monkeypatch.setattr(index_cli, "validate_cloud_runtime", lambda loaded: settings)
    monkeypatch.setattr(
        index_cli,
        "build_cloud_runtime",
        lambda checked: SimpleNamespace(
            settings=checked,
            embedding_provider=object(),
            vector_store=object(),
            keyword_search=object(),
        ),
    )

    class FakeCloudIndexer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build(self, *, recreate=False):
            assert recreate is True
            return CloudIndexStats(1, 2, 2, 2, 2, "manifest.json", "bm25.json")

    monkeypatch.setattr(index_cli, "CloudIndexer", FakeCloudIndexer)

    assert index_cli.main(["--recreate"]) == 0
    output = capsys.readouterr().out
    assert "Cloud 索引构建完成" in output
    assert "加载文档数：1" in output


def test_cloud_query_cli_defaults_to_agent(monkeypatch, capsys):
    settings = _settings()

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def ask(self, query, *, filters=None):
            self.calls.append((query, filters))
            return AgentRunResult(query=query, answer="Agent 回答", citations=[], trace=[], metrics={"ok": True})

    agent = FakeAgent()
    monkeypatch.setattr(
        query_cli,
        "build_cloud_agent_runtime",
        lambda loaded: SimpleNamespace(settings=settings, llm_agent=agent),
    )

    assert query_cli.main(["哪些岗位要求 RAG？", "--top-k", "3", "--source-type", "job", "--company", "Kairo"]) == 0
    output = capsys.readouterr().out

    assert "Agent 回答" in output
    assert agent.calls[0] == ("哪些岗位要求 RAG？", {"source_type": "job", "company": "Kairo"})
    assert settings.max_search_results == 3


def test_cloud_query_cli_baseline_uses_fixed_query_service(monkeypatch, capsys):
    settings = _settings()

    class FakeQueryService:
        def __init__(self):
            self.calls = []

        def ask(self, query, *, filters=None):
            self.calls.append((query, filters))
            return CloudQueryResult(query=query, answer="Baseline 回答", citations=[], trace=[], metrics={"ok": True})

    service = FakeQueryService()
    monkeypatch.setattr(
        query_cli,
        "build_cloud_query_runtime",
        lambda loaded: SimpleNamespace(settings=settings, query_service=service),
    )

    assert query_cli.main(["哪些岗位要求 RAG？", "--baseline"]) == 0
    output = capsys.readouterr().out

    assert "Baseline 回答" in output
    assert service.calls[0] == ("哪些岗位要求 RAG？", None)
