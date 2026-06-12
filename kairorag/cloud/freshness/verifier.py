"""云化岗位 freshness verifier。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from kairorag.cloud.freshness.schemas import JobFreshnessEvidence, JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk
from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.websearch import WebSearchProvider, WebSearchResult


ACTIVE_SIGNALS = [
    "apply",
    "apply now",
    "submit application",
    "job details",
    "currently hiring",
    "open position",
    "招聘中",
    "立即申请",
    "申请职位",
]

CLOSED_SIGNALS = [
    "no longer accepting applications",
    "job expired",
    "position closed",
    "this job is no longer available",
    "not found",
    "404",
    "职位已关闭",
    "已停止招聘",
    "岗位已下线",
]


class CloudJobFreshnessVerifier:
    """基于真实 WebSearchProvider 的岗位实时状态验证器。"""

    def __init__(self, settings: KairoCloudSettings, web_search_provider: WebSearchProvider) -> None:
        self.settings = settings
        self.web_search_provider = web_search_provider

    def verify_from_chunk(self, chunk: CloudReadChunk) -> JobFreshnessResult:
        metadata = {**chunk.metadata, "chunk_id": chunk.chunk_id}
        metadata.setdefault("title", chunk.title)
        return self.verify_from_metadata(metadata, chunk_id=chunk.chunk_id)

    def verify_from_metadata(self, metadata: dict[str, Any], *, chunk_id: str | None = None) -> JobFreshnessResult:
        checked_at = _now()
        company = _first_text(metadata, "company", "company_name", "employer")
        title = _first_text(metadata, "job_title", "title", "position")
        job_id = _first_text(metadata, "job_id", "external_id", "id")
        original_url = _first_text(metadata, "original_url", "job_url", "url", "source_url")
        location = _first_text(metadata, "location", "job_location")
        resolved_chunk_id = chunk_id or _first_text(metadata, "chunk_id")

        if metadata.get("duplicate_of") or _truthy(metadata.get("duplicate")):
            return JobFreshnessResult(
                job_id=job_id,
                chunk_id=resolved_chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                status="duplicate",
                confidence=0.95,
                evidence=[],
                recommended_action="mark_duplicate",
                reason="metadata 已标记 duplicate 或 duplicate_of，优先判定为重复岗位。",
                checked_at=checked_at,
            )

        query = _compose_search_query(
            query=_first_text(metadata, "freshness_query", "query"),
            company=company,
            title=title,
            original_url=original_url,
            location=location,
            job_id=job_id,
        )
        return self.verify_by_query(
            query,
            company=company,
            title=title,
            job_id=job_id,
            original_url=original_url,
            chunk_id=resolved_chunk_id,
        )

    def verify_by_query(
        self,
        query: str,
        *,
        company: str | None = None,
        title: str | None = None,
        job_id: str | None = None,
        original_url: str | None = None,
        chunk_id: str | None = None,
    ) -> JobFreshnessResult:
        if not self.settings.job_freshness_enabled:
            raise KairoProviderError("岗位 freshness verification 未启用，无法调用 verify_job_freshness。")

        checked_at = _now()
        search_query = _compose_search_query(
            query=query,
            company=company,
            title=title,
            original_url=original_url,
            location=None,
            job_id=job_id,
        )
        if not search_query:
            return _unknown_result(
                job_id=job_id,
                chunk_id=chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                checked_at=checked_at,
                reason="缺少公司、职位、原始 URL 或可搜索 query，无法验证实时状态。",
            )

        max_results = min(self.settings.job_verification_max_results, self.settings.web_search_max_results)
        results = self.web_search_provider.search(search_query, max_results=max_results)
        return self._decide(
            results,
            job_id=job_id,
            chunk_id=chunk_id,
            company=company,
            title=title,
            original_url=original_url,
            checked_at=checked_at,
        )

    def _decide(
        self,
        results: list[WebSearchResult],
        *,
        job_id: str | None,
        chunk_id: str | None,
        company: str | None,
        title: str | None,
        original_url: str | None,
        checked_at: str,
    ) -> JobFreshnessResult:
        original_domain = _domain(original_url)
        evidence: list[JobFreshnessEvidence] = []
        best_closed: tuple[float, JobFreshnessEvidence] | None = None
        best_active: tuple[float, JobFreshnessEvidence] | None = None
        best_updated: tuple[float, JobFreshnessEvidence, str] | None = None
        best_weak: tuple[float, JobFreshnessEvidence] | None = None

        for result in results:
            analysis = _analyze_result(
                result,
                company=company,
                title=title,
                original_url=original_url,
                original_domain=original_domain,
                observed_at=checked_at,
            )
            if analysis.evidence is not None:
                evidence.append(analysis.evidence)
            if analysis.closed and (best_closed is None or analysis.confidence > best_closed[0]):
                best_closed = (analysis.confidence, analysis.evidence)
            if analysis.updated_url and analysis.active and (best_updated is None or analysis.confidence > best_updated[0]):
                best_updated = (analysis.confidence, analysis.evidence, analysis.updated_url)
            if analysis.active and not analysis.updated_url and (
                best_active is None or analysis.confidence > best_active[0]
            ):
                best_active = (analysis.confidence, analysis.evidence)
            if analysis.weak and (best_weak is None or analysis.confidence > best_weak[0]):
                best_weak = (analysis.confidence, analysis.evidence)

        if best_closed is not None and best_closed[0] >= 0.5:
            return JobFreshnessResult(
                job_id=job_id,
                chunk_id=chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                status="closed",
                confidence=_cap(best_closed[0]),
                evidence=evidence,
                recommended_action="mark_closed",
                reason="搜索证据包含明确关闭、过期或不可用信号，因此判定岗位已关闭。",
                checked_at=checked_at,
            )
        if best_updated is not None and best_updated[0] >= 0.55:
            return JobFreshnessResult(
                job_id=job_id,
                chunk_id=chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                status="updated",
                confidence=_cap(best_updated[0]),
                evidence=evidence,
                recommended_action="update_url",
                reason="找到同公司或同职位的可申请新链接，且 URL 与原始链接不同。",
                checked_at=checked_at,
                new_url=best_updated[2],
            )
        if best_active is not None and best_active[0] >= 0.45:
            return JobFreshnessResult(
                job_id=job_id,
                chunk_id=chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                status="active",
                confidence=_cap(best_active[0]),
                evidence=evidence,
                recommended_action="keep_active",
                reason="搜索证据匹配岗位，并包含明确申请或开放信号，倾向判定为 active。",
                checked_at=checked_at,
            )
        if best_weak is not None:
            return JobFreshnessResult(
                job_id=job_id,
                chunk_id=chunk_id,
                company=company,
                title=title,
                original_url=original_url,
                status="stale",
                confidence=_cap(best_weak[0]),
                evidence=evidence,
                recommended_action="mark_stale",
                reason="找到相关网页线索，但没有明确 active 或 closed 信号，不能确认实时状态。",
                checked_at=checked_at,
            )
        return _unknown_result(
            job_id=job_id,
            chunk_id=chunk_id,
            company=company,
            title=title,
            original_url=original_url,
            checked_at=checked_at,
            reason="未找到足够匹配的网页证据，无法确认岗位实时状态。",
        )


class _ResultAnalysis:
    def __init__(
        self,
        *,
        evidence: JobFreshnessEvidence | None,
        confidence: float,
        active: bool = False,
        closed: bool = False,
        weak: bool = False,
        updated_url: str | None = None,
    ) -> None:
        self.evidence = evidence
        self.confidence = confidence
        self.active = active
        self.closed = closed
        self.weak = weak
        self.updated_url = updated_url


def _analyze_result(
    result: WebSearchResult,
    *,
    company: str | None,
    title: str | None,
    original_url: str | None,
    original_domain: str | None,
    observed_at: str,
) -> _ResultAnalysis:
    haystack = " ".join([result.title, result.snippet, result.url]).lower()
    company_match = _contains(haystack, company)
    title_match = _title_matches(haystack, title)
    domain_match = bool(original_domain and _domain(result.url) == original_domain)
    same_url = bool(original_url and _normalize_url(result.url) == _normalize_url(original_url))
    active_signal = _find_signal(haystack, ACTIVE_SIGNALS)
    closed_signal = _find_signal(haystack, CLOSED_SIGNALS)

    confidence = 0.0
    if company_match and title_match:
        confidence += 0.3
    elif company_match or title_match:
        confidence += 0.15
    if domain_match:
        confidence += 0.2
    if closed_signal:
        confidence += 0.5
        evidence = _evidence(result, f"closed:{closed_signal}", confidence, observed_at)
        return _ResultAnalysis(evidence=evidence, confidence=_cap(confidence), closed=True)
    if active_signal:
        confidence += 0.3
        updated_url = None
        if original_url and not same_url and (company_match or title_match):
            confidence += 0.4
            updated_url = result.url
            signal = f"updated_url_active:{active_signal}"
        else:
            signal = f"active:{active_signal}"
        evidence = _evidence(result, signal, confidence, observed_at)
        return _ResultAnalysis(
            evidence=evidence,
            confidence=_cap(confidence),
            active=True,
            updated_url=updated_url,
        )
    if company_match or title_match or domain_match:
        confidence = max(confidence, 0.2 if domain_match else 0.15)
        evidence = _evidence(result, "weak_match_no_status_signal", confidence, observed_at)
        return _ResultAnalysis(evidence=evidence, confidence=_cap(confidence), weak=True)
    return _ResultAnalysis(evidence=None, confidence=0.0)


def _evidence(result: WebSearchResult, signal: str, confidence: float, observed_at: str) -> JobFreshnessEvidence:
    return JobFreshnessEvidence(
        source=result.source,
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        signal=signal,
        confidence=_cap(confidence),
        observed_at=observed_at,
    )


def _unknown_result(
    *,
    job_id: str | None,
    chunk_id: str | None,
    company: str | None,
    title: str | None,
    original_url: str | None,
    checked_at: str,
    reason: str,
) -> JobFreshnessResult:
    return JobFreshnessResult(
        job_id=job_id,
        chunk_id=chunk_id,
        company=company,
        title=title,
        original_url=original_url,
        status="unknown",
        confidence=0.0,
        evidence=[],
        recommended_action="manual_review",
        reason=reason,
        checked_at=checked_at,
    )


def _compose_search_query(
    *,
    query: str | None,
    company: str | None,
    title: str | None,
    original_url: str | None,
    location: str | None,
    job_id: str | None,
) -> str:
    parts: list[str] = []
    for value in (company, title, _domain(original_url), location, job_id, query):
        clean = (value or "").strip()
        if clean and clean.lower() not in {part.lower() for part in parts}:
            parts.append(clean)
    return " ".join(parts)


def _first_text(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def _find_signal(text: str, signals: list[str]) -> str | None:
    for signal in signals:
        if signal.lower() in text:
            return signal
    return None


def _contains(text: str, needle: str | None) -> bool:
    return bool(needle and needle.strip().lower() in text)


def _title_matches(text: str, title: str | None) -> bool:
    if not title or not title.strip():
        return False
    lowered_title = title.strip().lower()
    if lowered_title in text:
        return True
    tokens = [token for token in _simple_tokens(lowered_title) if len(token) > 2]
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in text)
    return matched / len(tokens) >= 0.6


def _simple_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    return f"{netloc}{path}".lower()


def _cap(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
