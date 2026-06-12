from pydantic import SecretStr

from kairorag import cli
from kairorag.config import KairoCloudSettings


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("secret-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("secret-qdrant"),
        tavily_api_key=SecretStr("secret-tavily"),
    )


def test_cli_help_for_all_subcommands():
    for argv in (
        ["--help"],
        ["index", "--help"],
        ["query", "--help"],
        ["agent", "--help"],
        ["verify", "--help"],
        ["eval", "--help"],
        ["doctor", "--help"],
        ["config", "--help"],
    ):
        try:
            cli.main(argv)
        except SystemExit as exc:
            assert exc.code == 0


def test_cli_index_dispatches_cloud_index(monkeypatch):
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(cli.index_cli, "main", fake_main)

    assert cli.main(["index", "--recreate", "--batch-size", "8", "--manifest-path", "m.json"]) == 0

    assert called["argv"] == ["--recreate", "--batch-size", "8", "--manifest-path", "m.json"]


def test_cli_query_dispatches_cloud_query(monkeypatch):
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(cli.query_cli, "main", fake_main)

    assert (
        cli.main(
            [
                "query",
                "哪些岗位要求 RAG？",
                "--baseline",
                "--top-k",
                "3",
                "--source-type",
                "job",
                "--company",
                "Kairo",
                "--reranker",
                "cohere",
                "--rerank-top-k",
                "2",
                "--verify-freshness",
                "--show-verification",
                "--apply-freshness-update",
                "--dry-run",
                "false",
            ]
        )
        == 0
    )

    assert called["argv"][0] == "哪些岗位要求 RAG？"
    assert "--baseline" in called["argv"]
    assert "--reranker" in called["argv"]
    assert "false" in called["argv"]


def test_cli_agent_dispatches_agent_cli(monkeypatch):
    called = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(cli.agent_cli, "main", fake_main)

    assert cli.main(["agent", "请读取 chunk 后回答", "--max-tool-calls", "4", "--include-trace"]) == 0

    assert called["argv"] == ["请读取 chunk 后回答", "--include-trace", "--max-tool-calls", "4"]


def test_cli_eval_generates_files(tmp_path):
    assert cli.main(["eval", "--suite", "retrieval", "--output-dir", str(tmp_path), "--fake-providers"]) == 0

    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_dashboard.html").exists()


def test_cli_config_redacts_secrets(monkeypatch, capsys):
    monkeypatch.setattr(cli, "KairoCloudSettings", _settings)

    assert cli.main(["config", "--json", "--show-paths"]) == 0
    output = capsys.readouterr().out

    assert "secret-openai" not in output
    assert "secret-qdrant" not in output
    assert "has_openai_api_key" in output
