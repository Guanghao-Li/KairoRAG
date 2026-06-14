"""OpenAI live contract tests，默认跳过。"""

from __future__ import annotations

import pytest

from tests.live.helpers import require_env, requires_live


pytestmark = [pytest.mark.live, pytest.mark.external]


def test_live_openai_embedding_returns_vector():
    settings = requires_live("openai")
    api_key = require_env("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=settings.timeout_seconds)
    response = client.embeddings.create(model="text-embedding-3-small", input="OpenAI")
    vector = response.data[0].embedding

    assert isinstance(vector, list)
    assert len(vector) > 0


def test_live_openai_chat_returns_short_text():
    settings = requires_live("openai")
    api_key = require_env("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=settings.timeout_seconds)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Return only: ok"}],
        max_tokens=8,
    )

    assert response.choices[0].message.content
