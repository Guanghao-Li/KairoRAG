"""live provider contract tests 的共享 helper。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class LiveSettings:
    enabled: bool
    provider: str
    timeout_seconds: float


def get_live_settings() -> LiveSettings:
    return LiveSettings(
        enabled=os.getenv("RUN_LIVE_PROVIDER_TESTS") == "1",
        provider=os.getenv("LIVE_PROVIDER", "all").strip().lower() or "all",
        timeout_seconds=float(os.getenv("LIVE_TEST_TIMEOUT_SECONDS", "8")),
    )


def requires_live(provider_name: str) -> LiveSettings:
    settings = get_live_settings()
    provider = provider_name.strip().lower()
    if not settings.enabled:
        pytest.skip("live provider tests 默认跳过；设置 RUN_LIVE_PROVIDER_TESTS=1 后才运行。")
    if settings.provider not in {"all", provider}:
        pytest.skip(f"LIVE_PROVIDER={settings.provider}，跳过 {provider}。")
    return settings


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"缺少 {name}，跳过对应 live provider test。")
    return value
