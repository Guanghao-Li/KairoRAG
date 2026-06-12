"""Cloud query service。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kairorag.cloud.answering import CloudCitation, GroundedCloudAnswerGenerator
from kairorag.cloud.retriever import CloudRetriever, RetrievalTraceStep
from kairorag.config import KairoCloudSettings


@dataclass(frozen=True)
class CloudQueryResult:
    query: str
    answer: str
    citations: list[CloudCitation]
    trace: list[RetrievalTraceStep]
    metrics: dict[str, Any]


class CloudQueryService:
    """组合 cloud retriever 与 grounded answer generator。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        retriever: CloudRetriever,
        answer_generator: GroundedCloudAnswerGenerator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.answer_generator = answer_generator

    def ask(self, query: str, *, filters: dict[str, Any] | None = None) -> CloudQueryResult:
        """执行 cloud RAG 主链路并返回结构化结果。"""

        if not query or not query.strip():
            raise ValueError("query 不能为空。")
        retrieval_result = self.retriever.retrieve(query, filters=filters)
        answer = self.answer_generator.answer(query, retrieval_result)
        metrics = {
            **{f"retrieval_{key}": value for key, value in retrieval_result.metrics.items()},
            **{f"answer_{key}": value for key, value in answer.metrics.items()},
        }
        return CloudQueryResult(
            query=query,
            answer=answer.answer,
            citations=answer.citations,
            trace=answer.retrieval_trace,
            metrics=metrics,
        )
