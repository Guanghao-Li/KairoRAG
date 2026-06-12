from pathlib import Path


def test_docker_artifacts_exist():
    assert Path("Dockerfile").exists()
    assert Path("docker-compose.yml").exists()
    assert Path(".env.example").exists()
    assert Path("Makefile").exists()


def test_env_example_contains_placeholders_without_real_keys():
    text = Path(".env.example").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in text
    assert "QDRANT_URL=http://qdrant:6333" in text
    assert "COHERE_API_KEY=" in text
    assert "sk-" not in text


def test_makefile_contains_required_targets():
    text = Path("Makefile").read_text(encoding="utf-8")

    for target in ("install", "test", "lint", "ci", "index", "query", "doctor", "eval", "docker-build"):
        assert f"{target}:" in text
