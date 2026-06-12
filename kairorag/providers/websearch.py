"""Web search provider 接口和阶段一骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_at: str | None = None
    raw: dict[str, Any] | None = None


class WebSearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        ...


class ConfiguredWebSearchProvider:
    """真实 web search provider 的配置骨架，不提供 mock fallback。"""

    def __init__(self, provider_name: str, api_key: SecretStr | None, timeout_seconds: int) -> None:
        if provider_name not in {"tavily", "serpapi", "bing"}:
            raise KairoProviderError(f"不支持的 web search provider：{provider_name}")
        if not api_key or not api_key.get_secret_value().strip():
            raise KairoProviderError(f"无法初始化 {provider_name} web search provider：缺少对应 API key。")
        self.provider_name = provider_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        del query, max_results
        # TODO：阶段二接入 provider 对应的真实 HTTP API，并补齐响应解析和重试策略。
        raise KairoProviderError(
            f"{self.provider_name} web search 已完成配置校验，但真实 HTTP 调用将在后续阶段接入。"
        )


class TavilyWebSearchProvider(ConfiguredWebSearchProvider):
    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__("tavily", settings.tavily_api_key, settings.request_timeout_seconds)


class SerpAPIWebSearchProvider(ConfiguredWebSearchProvider):
    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__("serpapi", settings.serpapi_api_key, settings.request_timeout_seconds)


class BingWebSearchProvider(ConfiguredWebSearchProvider):
    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__("bing", settings.bing_api_key, settings.request_timeout_seconds)
