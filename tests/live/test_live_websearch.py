"""Web search live contract tests，默认跳过。"""

from __future__ import annotations

import os

import pytest

from kairorag.config import KairoCloudSettings
from kairorag.providers import build_web_search_provider
from tests.live.helpers import require_env, requires_live


pytestmark = [pytest.mark.live, pytest.mark.external]


def test_live_websearch_low_cost_query():
    provider_name = os.getenv("LIVE_PROVIDER", os.getenv("WEB_SEARCH_PROVIDER", "tavily")).strip().lower()
    if provider_name in {"all", "websearch"}:
        provider_name = os.getenv("WEB_SEARCH_PROVIDER", "tavily").strip().lower()
    requires_live(provider_name)
    if provider_name == "tavily":
        require_env("TAVILY_API_KEY")
    elif provider_name == "serpapi":
        require_env("SERPAPI_API_KEY")
    elif provider_name == "bing":
        require_env("BING_API_KEY")
    else:
        pytest.skip(f"不支持的 websearch live provider：{provider_name}")

    settings = KairoCloudSettings(web_search_provider=provider_name, job_freshness_enabled=True)
    provider = build_web_search_provider(settings)
    results = provider.search("OpenAI", max_results=1)

    assert len(results) <= 1
