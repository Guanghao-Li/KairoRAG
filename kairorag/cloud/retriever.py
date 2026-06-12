"""Cloud-native 检索与 chunk_read 主流程。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from kairorag.cloud.manifest import CloudIndexManifest, find_chunk
from kairorag.cloud.observability import TraceEvent, TraceLogger, now_iso
from kairorag.config import KairoCloudSettings
from kairorag.ingestion.metadata import estimate_tokens
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.embeddings import EmbeddingProvider
from kairorag.providers.keyword import KeywordSearchProvider
from kairorag.providers.rerankers import RerankCandidate, RerankerProvider
from kairorag.providers.vectorstores import VectorStoreProvider


@dataclass(frozen=True)
class CloudSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CloudReadChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalTraceStep:
    step: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    search_results: list[CloudSearchResult]
    read_chunks: list[CloudReadChunk]
    trace: list[RetrievalTraceStep]
    metrics: dict[str, Any]


class CloudRetriever:
    """组合 embedding、Qdrant、BM25 和 reranker 的稳定 cloud 检索器。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        keyword_search: KeywordSearchProvider,
        reranker: RerankerProvider,
        manifest: CloudIndexManifest,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.keyword_search_provider = keyword_search
        self.reranker = reranker
        self.manifest = manifest
        self._trace: list[RetrievalTraceStep] = []
        self.trace_logger = trace_logger or TraceLogger(
            settings.trace_log_path,
            enabled=settings.observability_enabled,
        )

    def semantic_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[CloudSearchResult]:
        """用 OpenAI query embedding 和 Qdrant 做语义检索。"""

        _validate_query(query)
        query_embedding = self.embedding_provider.embed_query(query)
        vector_results = self.vector_store.search(query_embedding.vector, top_k=top_k, filters=filters)
        results = [
            CloudSearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=str(result.metadata.get("title", result.doc_id)),
                text=result.text,
                score=result.score,
                source="semantic",
                metadata=dict(result.metadata),
            )
            for result in vector_results
        ]
        self._add_trace("semantic_search", {"chunk_ids": [result.chunk_id for result in results]})
        return results

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[CloudSearchResult]:
        """用 BM25 provider 做关键词检索。"""

        _validate_query(query)
        keyword_results = self.keyword_search_provider.search(query, top_k=top_k, filters=filters)
        results = [
            CloudSearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                text=result.text,
                score=result.score,
                source="keyword",
                metadata=dict(result.metadata),
            )
            for result in keyword_results
        ]
        self._add_trace("keyword_search", {"chunk_ids": [result.chunk_id for result in results]})
        return results

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[CloudSearchResult]:
        """用 RRF 融合语义检索与关键词检索。"""

        _validate_query(query)
        semantic_results = self.semantic_search(query, top_k=top_k, filters=filters)
        keyword_results = self.keyword_search(query, top_k=top_k, filters=filters)
        fused = _rrf_fuse(semantic_results, keyword_results, top_k=top_k)
        self._add_trace("hybrid_fusion", {"chunk_ids": [result.chunk_id for result in fused]})
        return fused

    def chunk_read(self, chunk_ids: list[str]) -> list[CloudReadChunk]:
        """从 manifest 读取完整 chunk 文本，找不到的 chunk 会被跳过。"""

        read_chunks: list[CloudReadChunk] = []
        missing: list[str] = []
        for chunk_id in chunk_ids:
            record = find_chunk(self.manifest, chunk_id)
            if record is None:
                missing.append(chunk_id)
                continue
            read_chunks.append(
                CloudReadChunk(
                    chunk_id=record.chunk_id,
                    doc_id=record.doc_id,
                    title=record.title,
                    text=record.text,
                    metadata=dict(record.metadata),
                )
            )
        self._add_trace(
            "chunk_read",
            {"chunk_ids": [chunk.chunk_id for chunk in read_chunks], "missing_chunk_ids": missing},
        )
        return read_chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """执行 hybrid search、rerank、chunk_read 并返回 trace 与 metrics。"""

        _validate_query(query)
        run_id = f"retrieval-{uuid4()}"
        self._trace = [RetrievalTraceStep("query", {"query": query})]
        self._log_event(run_id, "retrieval_started", query, {"filters": filters or {}})
        search_limit = min(
            top_k or self.settings.rerank_candidate_count,
            self.settings.rerank_candidate_count,
            self.settings.max_search_results,
        )
        search_results = self.hybrid_search(query, top_k=search_limit, filters=filters)
        candidates = [
            RerankCandidate(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata={**result.metadata, "source": result.source},
                base_score=result.score,
            )
            for result in search_results
        ]
        rerank_top_k = min(self.settings.rerank_top_k, self.settings.max_chunks_to_read, max(1, len(candidates)))
        rerank_started = time.perf_counter()
        self._log_event(
            run_id,
            "rerank_started",
            query,
            {
                "reranker_provider": self.settings.reranker_provider,
                "candidate_count": len(candidates),
                "top_k": rerank_top_k,
            },
        )
        try:
            reranked = self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        except KairoProviderError as exc:
            latency_ms = round((time.perf_counter() - rerank_started) * 1000, 3)
            self._add_trace(
                "rerank",
                {
                    "reranker_provider": self.settings.reranker_provider,
                    "rerank_candidate_count": len(candidates),
                    "rerank_output_count": 0,
                    "rerank_output_chunk_ids": [],
                    "rerank_latency_ms": latency_ms,
                    "error": str(exc),
                },
            )
            self._log_event(run_id, "provider_error", query, {"provider": self.settings.reranker_provider, "error": str(exc)})
            raise
        latency_ms = round((time.perf_counter() - rerank_started) * 1000, 3)
        reranked_ids = [result.chunk_id for result in reranked]
        rerank_scores = {result.chunk_id: result.score for result in reranked}
        self._add_trace(
            "rerank",
            {
                "reranker_provider": self.settings.reranker_provider,
                "rerank_candidate_count": len(candidates),
                "rerank_output_count": len(reranked),
                "rerank_output_chunk_ids": reranked_ids,
                "rerank_scores": rerank_scores,
                "rerank_latency_ms": latency_ms,
                "chunk_ids": reranked_ids,
            },
        )
        self._log_event(
            run_id,
            "rerank_finished",
            query,
            {
                "reranker_provider": self.settings.reranker_provider,
                "rerank_candidate_count": len(candidates),
                "rerank_output_count": len(reranked),
                "rerank_output_chunk_ids": reranked_ids,
                "rerank_latency_ms": latency_ms,
            },
        )
        if candidates and not reranked and self.settings.reranker_provider != "base_score":
            metrics = {
                "search_result_count": len(search_results),
                "read_chunk_count": 0,
                "tool_call_count": len([step for step in self._trace if step.step != "query"]),
                "context_token_estimate": 0,
                "rerank_enabled": True,
                "reranker_provider": self.settings.reranker_provider,
                "rerank_latency_ms": latency_ms,
                "rerank_candidate_count": len(candidates),
                "rerank_output_count": 0,
                "rerank_error": "reranker 返回空结果。",
            }
            self._log_event(run_id, "retrieval_finished", query, metrics)
            return RetrievalResult(
                query=query,
                search_results=search_results,
                read_chunks=[],
                trace=list(self._trace),
                metrics=metrics,
            )
        selected_ids, context_token_estimate = self._select_chunk_ids_with_budget(reranked_ids)
        read_chunks = self.chunk_read(selected_ids)
        metrics = {
            "search_result_count": len(search_results),
            "read_chunk_count": len(read_chunks),
            "tool_call_count": len([step for step in self._trace if step.step != "query"]),
            "context_token_estimate": context_token_estimate,
            "rerank_enabled": True,
            "reranker_provider": self.settings.reranker_provider,
            "rerank_latency_ms": latency_ms,
            "rerank_candidate_count": len(candidates),
            "rerank_output_count": len(reranked),
        }
        self._log_event(run_id, "retrieval_finished", query, metrics)
        return RetrievalResult(
            query=query,
            search_results=search_results,
            read_chunks=read_chunks,
            trace=list(self._trace),
            metrics=metrics,
        )

    def _select_chunk_ids_with_budget(self, chunk_ids: list[str]) -> tuple[list[str], int]:
        selected: list[str] = []
        total_tokens = 0
        for chunk_id in chunk_ids:
            record = find_chunk(self.manifest, chunk_id)
            if record is None:
                continue
            token_count = estimate_tokens(record.text)
            if selected and total_tokens + token_count > self.settings.max_context_tokens:
                continue
            selected.append(chunk_id)
            total_tokens += token_count
        return selected[: self.settings.max_chunks_to_read], total_tokens

    def _add_trace(self, step: str, detail: dict[str, Any]) -> None:
        self._trace.append(RetrievalTraceStep(step=step, detail=detail))

    def _log_event(self, run_id: str, event_type: str, query: str | None, detail: dict[str, Any]) -> None:
        self.trace_logger.log(
            TraceEvent(
                event_type=event_type,
                timestamp=now_iso(),
                run_id=run_id,
                query=query,
                detail=detail,
            )
        )


def _rrf_fuse(
    semantic_results: list[CloudSearchResult],
    keyword_results: list[CloudSearchResult],
    *,
    top_k: int,
    rrf_k: int = 60,
    keyword_weight: float = 1.0,
    semantic_weight: float = 0.9,
) -> list[CloudSearchResult]:
    by_chunk: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(semantic_results, start=1):
        item = by_chunk.setdefault(result.chunk_id, {"result": result, "score": 0.0, "metadata": dict(result.metadata)})
        item["score"] += semantic_weight / (rrf_k + rank)
        item["metadata"]["semantic_rank"] = rank
        item["metadata"]["semantic_score"] = result.score
    for rank, result in enumerate(keyword_results, start=1):
        item = by_chunk.setdefault(result.chunk_id, {"result": result, "score": 0.0, "metadata": dict(result.metadata)})
        item["score"] += keyword_weight / (rrf_k + rank)
        item["metadata"]["keyword_rank"] = rank
        item["metadata"]["keyword_score"] = result.score

    fused: list[CloudSearchResult] = []
    for item in by_chunk.values():
        result = item["result"]
        metadata = dict(item["metadata"])
        metadata["rrf_score"] = item["score"]
        fused.append(
            CloudSearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                text=result.text,
                score=float(item["score"]),
                source="hybrid",
                metadata=metadata,
            )
        )
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused[:top_k]


def _validate_query(query: str) -> None:
    if not query or not query.strip():
        raise ValueError("query 不能为空。")
