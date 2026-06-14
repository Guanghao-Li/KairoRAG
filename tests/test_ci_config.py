from pathlib import Path


def test_pre_commit_config_exists():
    text = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "ruff" in text
    assert "detect-private-key" in text
    assert "check-added-large-files" in text


def test_live_workflow_is_manual_only():
    text = Path(".github/workflows/live-contract-tests.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text


def test_ci_does_not_require_real_keys():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" not in text
    assert "RUN_LIVE_PROVIDER_TESTS" not in text
    assert "coverage" in text
