from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import LLMMessage, OpenAILLMProvider


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


class FakeCompletions:
    def __init__(self):
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="测试回答"))],
            model=payload["model"],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model_dump=lambda: {"id": "fake-response", "model": payload["model"]},
        )


class ErrorCompletions:
    def create(self, **payload):
        del payload
        raise RuntimeError("上游失败")


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def test_openai_llm_provider_returns_content_and_usage():
    completions = FakeCompletions()
    provider = OpenAILLMProvider(_settings(), client=FakeClient(completions))

    response = provider.complete(
        [LLMMessage(role="user", content="你好")],
        response_format={"type": "json_object"},
    )

    assert response.content == "测试回答"
    assert response.usage["total_tokens"] == 5
    assert completions.payload["response_format"] == {"type": "json_object"}


def test_openai_llm_provider_wraps_api_error():
    provider = OpenAILLMProvider(_settings(), client=FakeClient(ErrorCompletions()))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="你好")])

    assert "OpenAI LLM 调用失败" in str(exc_info.value)
