from pathlib import Path

from kairorag.release.hygiene import REQUIRED_GITIGNORE_RULES, RepoHygieneChecker


def _write_minimal_repo(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    (root / "kairorag" / "cloud").mkdir(parents=True)
    (root / "kairorag" / "cloud" / "query.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("只推荐 kairo CLI。\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (root / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    example_key_name = "OPENAI" + "_API_KEY"
    (root / ".env.example").write_text(f"{example_key_name}=\n", encoding="utf-8")
    (root / "Makefile").write_text("test:\n\tpython -m pytest\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project.scripts]\nkairo = "kairorag.cli:main"\n',
        encoding="utf-8",
    )
    (root / ".gitignore").write_text("\n".join(REQUIRED_GITIGNORE_RULES) + "\n", encoding="utf-8")


def test_repo_hygiene_detects_env_and_generated_files(tmp_path):
    _write_minimal_repo(tmp_path)
    fake_key = "sk-" + "real-looking-value-123456"
    key_name = "OPENAI" + "_API_KEY"
    (tmp_path / ".env").write_text(f"{key_name}={fake_key}\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "trace_events.jsonl").write_text("{}\n", encoding="utf-8")

    report = RepoHygieneChecker(tmp_path).run()
    codes = {issue.code for issue in report.issues}

    assert report.ok is False
    assert "env_file_present" in codes
    assert "possible_secret_assignment" in codes
    assert "generated_file_present" in codes


def test_repo_hygiene_required_gitignore_rules_are_present():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for rule in REQUIRED_GITIGNORE_RULES:
        assert rule in gitignore
