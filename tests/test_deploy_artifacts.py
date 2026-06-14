from pathlib import Path


def test_docker_compose_profiles_and_healthchecks_exist():
    text = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "profiles" in text
    assert "dev" in text
    assert "eval" in text
    assert "app" in text
    assert "healthcheck" in text


def test_deploy_docs_and_make_targets_exist():
    assert Path("deploy/README.md").exists()
    assert Path("deploy/systemd/kairo-ui.service.example").exists()
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in ("ui", "doctor-release", "eval-compare", "live-test", "docker-ui", "docker-eval"):
        assert f"{target}:" in makefile
