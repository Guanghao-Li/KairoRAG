"""Reranker live contract tests，默认跳过。"""

from __future__ import annotations

import os

import pytest

from kairorag.config import KairoCloudSettings
from kairorag.providers import build_reranker_provider
from tests.live.helpers import require_env, requires_live


pytestmark = [pytest.mark.live, pytest.mark.external]


def test_live_reranker_two_short_documents():
    provider_name = os.getenv("LIVE_PROVIDER", os.getenv("RERANKER_PROVIDER", "cohere")).strip().lower()
    if provider_name in {"all", "reranker"}:
        provider_name = os.getenv("RERANKER_PROVIDER", "cohere").strip().lower()
    requires_live(provider_name)
    if provider_name == "cohere":
        require_env("COHERE_API_KEY")
    elif provider_name == "jina":
        require_env("JINA_API_KEY")
    elif provider_name == "voyage":
        require_env("VOYAGE_API_KEY")
    elif provider_name == "openai_listwise":
        require_env("OPENAI_API_KEY")
    else:
        pytest.skip(f"不支持的 reranker live provider：{provider_name}")

    settings = KairoCloudSettings(reranker_provider=provider_name, rerank_timeout_seconds=8)
    reranker = build_reranker_provider(settings)
    results = reranker.rerank(
        "OpenAI",
        [
            {"chunk_id": "a", "text": "OpenAI builds AI models.", "score": 0.1},
            {"chunk_id": "b", "text": "A recipe for soup.", "score": 0.1},
        ],
        top_k=2,
    )

    assert len(results) == 2
