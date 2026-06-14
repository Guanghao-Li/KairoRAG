"""Qdrant live contract test，默认跳过。"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.live.helpers import requires_live


pytestmark = [pytest.mark.live, pytest.mark.external]


def test_live_qdrant_healthcheck():
    settings = requires_live("qdrant")
    url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    headers = {}
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if api_key:
        headers["api-key"] = api_key

    response = httpx.get(f"{url}/healthz", headers=headers, timeout=settings.timeout_seconds)

    assert response.status_code < 500
