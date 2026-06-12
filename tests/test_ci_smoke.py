from pathlib import Path


def test_ci_workflow_contains_required_checks():
    workflow = Path(".github/workflows/ci.yml")
    text = workflow.read_text(encoding="utf-8")

    assert workflow.exists()
    assert "ruff check" in text
    assert "python -m pytest" in text
    assert "Secret scan" in text
    assert "OPENAI_API_KEY" in text
    assert "python -m kairorag.cli --help" in text


def test_ci_workflow_does_not_require_real_secrets():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "secrets.OPENAI_API_KEY" not in text
    assert "secrets.QDRANT_API_KEY" not in text
