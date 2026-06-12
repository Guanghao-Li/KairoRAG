from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import LLMMessage, LLMToolSpec, OpenAILLMProvider


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
    )


class FakeCompletions:
    def __init__(self, message=None):
        self.payload = None
        self.message = message

    def create(self, **payload):
        self.payload = payload
        message = self.message or SimpleNamespace(content="测试回答")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
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


def _tool_spec():
    return LLMToolSpec(
        name="hybrid_search",
        description="检索",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )


def _tool_call(name="hybrid_search", arguments='{"query":"RAG"}', call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_openai_llm_provider_tool_calling_returns_one_tool_call():
    message = SimpleNamespace(content="", tool_calls=[_tool_call()])
    completions = FakeCompletions(message=message)
    provider = OpenAILLMProvider(_settings(), client=FakeClient(completions))

    response = provider.complete_with_tools([LLMMessage(role="user", content="规划")], [_tool_spec()])

    assert response.tool_calls[0].name == "hybrid_search"
    assert response.tool_calls[0].arguments == {"query": "RAG"}
    assert completions.payload["tools"][0]["function"]["name"] == "hybrid_search"


def test_openai_llm_provider_tool_calling_returns_multiple_tool_calls():
    message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call(name="keyword_search", arguments='{"query":"BM25"}', call_id="call-1"),
            _tool_call(name="semantic_search", arguments='{"query":"Qdrant"}', call_id="call-2"),
        ],
    )
    provider = OpenAILLMProvider(_settings(), client=FakeClient(FakeCompletions(message=message)))

    response = provider.complete_with_tools([LLMMessage(role="user", content="规划")], [_tool_spec()])

    assert [tool_call.name for tool_call in response.tool_calls] == ["keyword_search", "semantic_search"]


def test_openai_llm_provider_tool_calling_supports_no_tool_calls():
    message = SimpleNamespace(content="无需工具", tool_calls=[])
    provider = OpenAILLMProvider(_settings(), client=FakeClient(FakeCompletions(message=message)))

    response = provider.complete_with_tools([LLMMessage(role="user", content="规划")], [_tool_spec()])

    assert response.content == "无需工具"
    assert response.tool_calls == []


def test_openai_llm_provider_tool_calling_rejects_invalid_arguments_json():
    message = SimpleNamespace(content="", tool_calls=[_tool_call(arguments="{bad json")])
    provider = OpenAILLMProvider(_settings(), client=FakeClient(FakeCompletions(message=message)))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.complete_with_tools([LLMMessage(role="user", content="规划")], [_tool_spec()])

    assert "arguments 不是合法 JSON" in str(exc_info.value)


def test_openai_llm_provider_tool_calling_wraps_api_error():
    provider = OpenAILLMProvider(_settings(), client=FakeClient(ErrorCompletions()))

    with pytest.raises(KairoProviderError) as exc_info:
        provider.complete_with_tools([LLMMessage(role="user", content="规划")], [_tool_spec()])

    assert "OpenAI tool calling 调用失败" in str(exc_info.value)
