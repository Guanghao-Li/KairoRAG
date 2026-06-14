import pytest

from tests.live.helpers import get_live_settings, requires_live


def test_live_tests_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RUN_LIVE_PROVIDER_TESTS", raising=False)

    settings = get_live_settings()

    assert settings.enabled is False
    with pytest.raises(pytest.skip.Exception):
        requires_live("openai")


def test_live_provider_selection(monkeypatch):
    monkeypatch.setenv("RUN_LIVE_PROVIDER_TESTS", "1")
    monkeypatch.setenv("LIVE_PROVIDER", "openai")

    assert requires_live("openai").provider == "openai"
    with pytest.raises(pytest.skip.Exception):
        requires_live("qdrant")
