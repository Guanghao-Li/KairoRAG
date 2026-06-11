"""Mockable web search abstraction for job verification."""

from __future__ import annotations

from kairorag.schemas import WebSearchResult


def search_web(query: str, top_k: int = 5) -> list[WebSearchResult]:
    """Return mock web evidence unless a real provider is added later."""

    lowered = query.lower()
    results: list[WebSearchResult] = []
    if "deltaops" in lowered:
        results.append(
            WebSearchResult(
                title="DeltaOps AI Knowledge Maintenance Engineer",
                url="mock://jobs/job_008_active_new",
                snippet="Apply now for the AI Knowledge Maintenance Engineer role at DeltaOps.",
                source="mock",
                score=0.95,
            )
        )
    if "northstar" in lowered:
        results.append(
            WebSearchResult(
                title="Northstar AI Agent RAG Engineer",
                url="mock://jobs/job_001_active",
                snippet="Submit application for AI Agent RAG Engineer.",
                source="mock",
                score=0.92,
            )
        )
    if "quietcloud" in lowered:
        results.append(
            WebSearchResult(
                title="QuietCloud careers",
                url="mock://jobs/job_005_unknown",
                snippet="The posting may have moved; no current application button found.",
                source="mock",
                score=0.42,
            )
        )
    if "vectorworks" in lowered:
        results.append(
            WebSearchResult(
                title="VectorWorks Labs posting closed",
                url="mock://jobs/job_003_closed",
                snippet="This job is no longer available.",
                source="mock",
                score=0.9,
            )
        )
    if not results:
        results.append(
            WebSearchResult(
                title="No reliable current posting found",
                url="mock://jobs/unknown",
                snippet="Unable to confirm current job status from mock search.",
                source="mock",
                score=0.2,
            )
        )
    return results[:top_k]

