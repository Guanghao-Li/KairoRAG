"""Reranker provider 接口和阶段一骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    base_score: float | None = None


@dataclass(frozen=True)
class RerankResult:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class RerankerProvider(Protocol):
    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        ...


class BaseScoreRerankerProvider:
    """阶段一默认重排器：仅按候选已有分数排序，不伪装成 cross-encoder。"""

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        del query
        ranked = sorted(candidates, key=lambda item: item.base_score or 0.0, reverse=True)
        return [
            RerankResult(
                chunk_id=candidate.chunk_id,
                score=float(candidate.base_score or 0.0),
                metadata=candidate.metadata,
            )
            for candidate in ranked[:top_k]
        ]


class CrossEncoderRerankerProvider:
    """cross-encoder reranker 的真实接入骨架。"""

    def __init__(self, settings: KairoCloudSettings) -> None:
        if not settings.cross_encoder_model:
            raise KairoProviderError(
                "无法启用 cross_encoder reranker：缺少 CROSS_ENCODER_MODEL；"
                "阶段一不会提供假的 cross-encoder fallback。"
            )
        self.model_name = settings.cross_encoder_model

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        del query, candidates, top_k
        raise NotImplementedError("cross_encoder reranker 仍是阶段一骨架，后续阶段会接入真实模型或云 rerank API。")
