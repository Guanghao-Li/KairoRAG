"""真实 Web Search provider 实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
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


class _HTTPWebSearchProvider:
    """统一处理密钥、超时、HTTP 错误和基础参数校验。"""

    provider_name: str

    def __init__(self, settings: KairoCloudSettings, api_key: SecretStr | None) -> None:
        if not api_key or not api_key.get_secret_value().strip():
            raise KairoProviderError(f"无法初始化 {self.provider_name} web search provider：缺少对应 API key。")
        self.settings = settings
        self.api_key = api_key
        self.timeout_seconds = settings.web_search_timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        raise NotImplementedError

    def _validate_search_args(self, query: str, max_results: int) -> str:
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("web search query 不能为空。")
        if max_results <= 0:
            raise ValueError("web search max_results 必须大于 0。")
        return clean_query

    def _post_json(self, url: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=json_body)
            return self._parse_response(response)
        except httpx.TimeoutException as exc:
            raise KairoProviderError(f"{self.provider_name} web search 请求超时。") from exc
        except httpx.HTTPError as exc:
            raise KairoProviderError(f"{self.provider_name} web search HTTP 请求失败。") from exc

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params, headers=headers)
            return self._parse_response(response)
        except httpx.TimeoutException as exc:
            raise KairoProviderError(f"{self.provider_name} web search 请求超时。") from exc
        except httpx.HTTPError as exc:
            raise KairoProviderError(f"{self.provider_name} web search HTTP 请求失败。") from exc

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise KairoProviderError(f"{self.provider_name} web search 返回 HTTP {status_code}。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise KairoProviderError(f"{self.provider_name} web search 返回内容不是合法 JSON。") from exc
        if not isinstance(payload, dict):
            raise KairoProviderError(f"{self.provider_name} web search 返回 JSON 顶层必须是 object。")
        return payload

    def _require_list(self, payload: dict[str, Any], field_name: str) -> list[Any]:
        value = payload.get(field_name)
        if value is None:
            raise KairoProviderError(f"{self.provider_name} web search 响应缺少字段：{field_name}。")
        if not isinstance(value, list):
            raise KairoProviderError(f"{self.provider_name} web search 字段 {field_name} 必须是列表。")
        return value

    def _require_text(self, item: dict[str, Any], field_names: tuple[str, ...]) -> str:
        for field_name in field_names:
            value = item.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        joined = " / ".join(field_names)
        raise KairoProviderError(f"{self.provider_name} web search 单条结果缺少字段：{joined}。")

    def _optional_text(self, item: dict[str, Any], field_names: tuple[str, ...]) -> str | None:
        for field_name in field_names:
            value = item.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


class TavilySearchProvider(_HTTPWebSearchProvider):
    provider_name = "tavily"

    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__(settings, settings.tavily_api_key)

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        clean_query = self._validate_search_args(query, max_results)
        payload = self._post_json(
            "https://api.tavily.com/search",
            json_body={
                "api_key": self.api_key.get_secret_value(),
                "query": clean_query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        items = self._require_list(payload, "results")
        return [self._convert_item(item) for item in items if isinstance(item, dict)]

    def _convert_item(self, item: dict[str, Any]) -> WebSearchResult:
        return WebSearchResult(
            title=self._require_text(item, ("title",)),
            url=self._require_text(item, ("url",)),
            snippet=self._require_text(item, ("content", "snippet")),
            source=self.provider_name,
            published_at=self._optional_text(item, ("published_date", "published_at")),
            raw=_sanitize_raw(item),
        )


class SerpAPISearchProvider(_HTTPWebSearchProvider):
    provider_name = "serpapi"

    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__(settings, settings.serpapi_api_key)

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        clean_query = self._validate_search_args(query, max_results)
        payload = self._get_json(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": clean_query,
                "api_key": self.api_key.get_secret_value(),
                "num": max_results,
            },
        )
        items = self._require_list(payload, "organic_results")
        return [self._convert_item(item) for item in items if isinstance(item, dict)]

    def _convert_item(self, item: dict[str, Any]) -> WebSearchResult:
        return WebSearchResult(
            title=self._require_text(item, ("title",)),
            url=self._require_text(item, ("link", "url")),
            snippet=self._require_text(item, ("snippet",)),
            source=self.provider_name,
            published_at=self._optional_text(item, ("date", "published_at")),
            raw=_sanitize_raw(item),
        )


class BingSearchProvider(_HTTPWebSearchProvider):
    provider_name = "bing"

    def __init__(self, settings: KairoCloudSettings) -> None:
        super().__init__(settings, settings.bing_api_key)

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        clean_query = self._validate_search_args(query, max_results)
        payload = self._get_json(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": clean_query, "count": max_results},
            headers={"Ocp-Apim-Subscription-Key": self.api_key.get_secret_value()},
        )
        web_pages = payload.get("webPages")
        if web_pages is None:
            return []
        if not isinstance(web_pages, dict):
            raise KairoProviderError("bing web search 响应字段 webPages 必须是 object。")
        items = web_pages.get("value", [])
        if items is None:
            return []
        if not isinstance(items, list):
            raise KairoProviderError("bing web search 响应字段 webPages.value 必须是列表。")
        return [self._convert_item(item) for item in items if isinstance(item, dict)]

    def _convert_item(self, item: dict[str, Any]) -> WebSearchResult:
        return WebSearchResult(
            title=self._require_text(item, ("name", "title")),
            url=self._require_text(item, ("url",)),
            snippet=self._require_text(item, ("snippet",)),
            source=self.provider_name,
            published_at=self._optional_text(item, ("dateLastCrawled", "published_at")),
            raw=_sanitize_raw(item),
        )


def _sanitize_raw(value: dict[str, Any]) -> dict[str, Any]:
    """递归删除可能包含密钥的字段，避免 trace 或结果中泄露敏感信息。"""

    safe: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in ("key", "token", "secret", "subscription")):
            continue
        if isinstance(item, dict):
            safe[key] = _sanitize_raw(item)
        elif isinstance(item, list):
            safe[key] = [_sanitize_raw(child) if isinstance(child, dict) else child for child in item]
        else:
            safe[key] = item
    return safe


TavilyWebSearchProvider = TavilySearchProvider
SerpAPIWebSearchProvider = SerpAPISearchProvider
BingWebSearchProvider = BingSearchProvider

