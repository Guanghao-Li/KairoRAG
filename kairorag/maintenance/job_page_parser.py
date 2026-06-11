"""Parse job pages or mock URLs into evidence signals."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from kairorag.schemas import JobPageEvidence


ACTIVE_PHRASES = ["Apply now", "Submit application", "正在招聘", "立即申请", "投递"]
CLOSED_PHRASES = [
    "This job is no longer available",
    "Job closed",
    "Position filled",
    "职位已关闭",
    "职位已下线",
    "招聘已结束",
]


MOCK_PAGES: dict[str, tuple[int, str]] = {
    "mock://jobs/job_001_active": (200, "Northstar AI Agent RAG Engineer Remote US Apply now"),
    "mock://jobs/job_002_active": (200, "HelioData LLM Knowledge Base Intern Submit application"),
    "mock://jobs/job_003_closed": (200, "VectorWorks Labs Agentic Retrieval Research Assistant This job is no longer available Position filled"),
    "mock://jobs/job_004_stale": (404, "Atlas Robotics Multi-Agent Platform Engineer page not found"),
    "mock://jobs/job_005_unknown": (200, "QuietCloud careers role may have moved"),
    "mock://jobs/job_007_active": (200, "CivicFlow AI RAG Evaluation Engineer Washington DC Apply now"),
    "mock://jobs/job_008_updated": (200, "DeltaOps old posting moved to new careers page"),
    "mock://jobs/job_008_active_new": (200, "DeltaOps AI Knowledge Maintenance Engineer Austin TX Apply now"),
    "mock://jobs/unknown": (404, "No reliable current posting found"),
}


def _detect_status(text: str) -> tuple[str, list[str]]:
    detected: list[str] = []
    for phrase in ACTIVE_PHRASES:
        if phrase.lower() in text.lower():
            detected.append(phrase)
    for phrase in CLOSED_PHRASES:
        if phrase.lower() in text.lower():
            detected.append(phrase)
    if any(phrase in ACTIVE_PHRASES for phrase in detected):
        return "active", detected
    if any(phrase in CLOSED_PHRASES for phrase in detected):
        return "closed", detected
    return "unknown", detected


def _match(text: str, value: str) -> bool:
    if not value:
        return False
    return bool(re.search(re.escape(value), text, flags=re.IGNORECASE))


def parse_job_page(
    url: str,
    expected_title: str = "",
    expected_company: str = "",
    expected_location: str = "",
    timeout: float = 5.0,
    retries: int = 1,
) -> JobPageEvidence:
    """Parse a job page with timeout/retry, or deterministic mock URL."""

    if url.startswith("mock://"):
        status_code, text = MOCK_PAGES.get(url, (404, "No mock page"))
        status, phrases = _detect_status(text)
        if status_code >= 400 and status == "unknown":
            error = f"mock_http_{status_code}"
        else:
            error = ""
        return JobPageEvidence(
            url=url,
            status_signal=status,
            title_match=_match(text, expected_title),
            company_match=_match(text, expected_company),
            location_match=_match(text, expected_location),
            snippet=text[:280],
            detected_phrases=phrases,
            http_status=status_code,
            error=error,
        )

    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional dependency branch
        return JobPageEvidence(url=url, status_signal="unknown", error=f"missing_web_dependency:{exc}")

    headers = {"User-Agent": "KairoRAG/0.1 job freshness verifier"}
    last_error = ""
    for _attempt in range(max(1, retries)):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(" ", strip=True)
            status, phrases = _detect_status(text)
            return JobPageEvidence(
                url=url,
                status_signal=status,
                title_match=_match(text, expected_title),
                company_match=_match(text, expected_company) or _match(urlparse(url).netloc, expected_company),
                location_match=_match(text, expected_location),
                snippet=text[:280],
                detected_phrases=phrases,
                http_status=response.status_code,
                error="" if response.ok else f"http_{response.status_code}",
            )
        except Exception as exc:  # pragma: no cover - network branch
            last_error = str(exc)
    return JobPageEvidence(url=url, status_signal="unknown", error=last_error)
