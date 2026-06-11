"""Context budget controls for agentic retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from kairorag.ingestion.metadata import estimate_tokens
from kairorag.schemas import BudgetReport, ReadChunkResult, SearchResult


@dataclass
class ContextBudgetManager:
    """Track search, read, token, and tool-call budgets."""

    max_search_results: int = 10
    max_chunks_to_read: int = 5
    max_context_tokens: int = 1800
    max_tool_calls: int = 12
    deduplicate_same_chunk: bool = True
    avoid_reading_same_chunk_twice: bool = True
    retrieved_chunks: list[str] = field(default_factory=list)
    read_chunks: list[str] = field(default_factory=list)
    estimated_context_tokens: int = 0
    dropped_chunks: list[str] = field(default_factory=list)
    tool_call_count: int = 0

    def record_tool_call(self, name: str) -> None:
        self.tool_call_count += 1

    def can_call_tool(self) -> bool:
        return self.tool_call_count < self.max_tool_calls

    def limit_search_results(self, results: list[SearchResult]) -> list[SearchResult]:
        limited: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            if self.deduplicate_same_chunk and result.chunk_id in seen:
                self.dropped_chunks.append(result.chunk_id)
                continue
            seen.add(result.chunk_id)
            if len(limited) >= self.max_search_results:
                self.dropped_chunks.append(result.chunk_id)
                continue
            limited.append(result)
            self.retrieved_chunks.append(result.chunk_id)
        return limited

    def can_read(self, chunk_id: str) -> bool:
        if len(self.read_chunks) >= self.max_chunks_to_read:
            self.dropped_chunks.append(chunk_id)
            return False
        if self.avoid_reading_same_chunk_twice and chunk_id in self.read_chunks:
            self.dropped_chunks.append(chunk_id)
            return False
        return True

    def record_read(self, chunk: ReadChunkResult) -> bool:
        tokens = estimate_tokens(chunk.text)
        if self.estimated_context_tokens + tokens > self.max_context_tokens:
            self.dropped_chunks.append(chunk.chunk_id)
            return False
        self.read_chunks.append(chunk.chunk_id)
        self.estimated_context_tokens += tokens
        return True

    def report(self) -> BudgetReport:
        return BudgetReport(
            retrieved_chunks=self.retrieved_chunks,
            read_chunks=self.read_chunks,
            estimated_context_tokens=self.estimated_context_tokens,
            dropped_chunks=self.dropped_chunks,
        )

