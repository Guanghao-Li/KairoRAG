"""LLM provider 接口、tool calling 数据结构和 OpenAI 实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class LLMToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMToolResult:
    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any]
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSpec],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...


class OpenAILLMProvider:
    """使用 OpenAI 官方 SDK 的真实 LLM provider。"""

    def __init__(self, settings: KairoCloudSettings, client: Any | None = None) -> None:
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
            raise KairoProviderError("无法初始化 OpenAI LLM provider：缺少 OPENAI_API_KEY。")
        self.settings = settings
        self.model = settings.openai_chat_model
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - 依赖缺失分支
            raise KairoProviderError("无法初始化 OpenAI LLM provider：缺少 openai 依赖。") from exc
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("LLM messages 不能为空。")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**payload)
            choice = response.choices[0]
            content = getattr(choice.message, "content", "") or ""
            return LLMResponse(
                content=content,
                model=getattr(response, "model", self.model),
                usage=_model_dump(getattr(response, "usage", {})),
                raw=_model_dump(response),
            )
        except Exception as exc:
            raise KairoProviderError(f"OpenAI LLM 调用失败：{exc}") from exc

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSpec],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("LLM messages 不能为空。")
        if not tools:
            raise ValueError("LLM tools 不能为空。")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
            "tools": [_tool_spec_payload(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": temperature,
        }
        try:
            response = self.client.chat.completions.create(**payload)
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", "") or ""
            tool_calls = [_parse_tool_call(tool_call) for tool_call in getattr(message, "tool_calls", []) or []]
            return LLMResponse(
                content=content,
                model=getattr(response, "model", self.model),
                usage=_model_dump(getattr(response, "usage", {})),
                tool_calls=tool_calls,
                raw=_model_dump(response),
            )
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"OpenAI tool calling 调用失败：{exc}") from exc


def _message_payload(message: LLMMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _tool_spec_payload(tool: LLMToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_tool_call(tool_call: Any) -> LLMToolCall:
    function = getattr(tool_call, "function", None)
    name = str(getattr(function, "name", "") or "")
    raw_arguments = getattr(function, "arguments", "{}")
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        raise KairoProviderError(f"OpenAI tool call arguments 不是合法 JSON：tool={name}") from exc
    if not isinstance(arguments, dict):
        raise KairoProviderError(f"OpenAI tool call arguments 必须是 JSON object：tool={name}")
    return LLMToolCall(
        id=str(getattr(tool_call, "id", "") or ""),
        name=name,
        arguments=arguments,
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}
