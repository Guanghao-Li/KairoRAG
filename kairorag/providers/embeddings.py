"""Embedding provider 接口和 OpenAI 实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import _model_dump


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    model: str
    usage: dict[str, Any] | None = None


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        ...

    def embed_query(self, query: str) -> EmbeddingResult:
        ...


class OpenAIEmbeddingProvider:
    """使用 OpenAI 官方 SDK 的真实 embedding provider。"""

    def __init__(self, settings: KairoCloudSettings, client: Any | None = None) -> None:
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
            raise KairoProviderError("无法初始化 OpenAI embedding provider：缺少 OPENAI_API_KEY。")
        self.settings = settings
        self.model = settings.openai_embedding_model
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - 依赖缺失分支
            raise KairoProviderError("无法初始化 OpenAI embedding provider：缺少 openai 依赖。") from exc
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        blank_positions = [index for index, text in enumerate(texts) if not text or not text.strip()]
        if blank_positions:
            raise ValueError(f"embedding 输入包含空字符串，位置：{blank_positions}")
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            usage = _model_dump(getattr(response, "usage", {}))
            rows = sorted(
                list(getattr(response, "data", [])),
                key=lambda item: getattr(item, "index", 0),
            )
            return [
                EmbeddingResult(
                    text=text,
                    vector=[float(value) for value in getattr(row, "embedding")],
                    model=getattr(response, "model", self.model),
                    usage=usage,
                )
                for text, row in zip(texts, rows)
            ]
        except Exception as exc:
            raise KairoProviderError(f"OpenAI embedding 调用失败：{exc}") from exc

    def embed_query(self, query: str) -> EmbeddingResult:
        if not query or not query.strip():
            raise ValueError("embedding query 不能为空字符串。")
        return self.embed_texts([query])[0]
