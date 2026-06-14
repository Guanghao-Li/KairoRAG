from pathlib import Path

from kairorag.release.checklist import build_release_checklist


def test_release_checklist_exists_and_mentions_required_steps():
    text = Path("RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    generated = build_release_checklist()

    for keyword in ("pytest", "ruff", "secret scan", "doctor", "eval", "docker", ".env", "版本号"):
        assert keyword in text
    assert "live tests 默认跳过" in text
    assert "legacy 主路径" in generated
