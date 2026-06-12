"""LLM provider 接口和 OpenAI 实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any]
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
            "messages": [{"role": message.role, "content": message.content} for message in messages],
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
