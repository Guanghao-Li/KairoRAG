"""Tool registry used by the KairoRAG agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kairorag.maintenance.deduplicator import detect_duplicate_jobs
from kairorag.maintenance.index_refresher import refresh_index_for_jobs
from kairorag.maintenance.job_verifier import verify_job
from kairorag.maintenance.knowledge_base_updater import apply_verification_result
from kairorag.retrieval.chunk_read import chunk_read
from kairorag.retrieval.context_compressor import compress_context
from kairorag.retrieval.hybrid_search import hybrid_search
from kairorag.retrieval.keyword_search import keyword_search
from kairorag.retrieval.reranker import rerank_context
from kairorag.retrieval.semantic_search import semantic_search


class ToolRegistry:
    """Minimal callable registry for retrieval and maintenance tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self._tools[name] = func

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name](**kwargs)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)


def default_tool_registry() -> ToolRegistry:
    """Register all tools required by the project spec."""

    registry = ToolRegistry()
    registry.register("keyword_search", keyword_search)
    registry.register("semantic_search", semantic_search)
    registry.register("hybrid_search", hybrid_search)
    registry.register("chunk_read", chunk_read)
    registry.register("rerank_context", rerank_context)
    registry.register("compress_context", compress_context)
    registry.register("verify_job_status_tool", verify_job)
    registry.register("update_knowledge_base_tool", apply_verification_result)
    registry.register("refresh_index_tool", refresh_index_for_jobs)
    registry.register("detect_duplicate_jobs_tool", detect_duplicate_jobs)
    return registry

