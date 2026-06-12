"""Shared serializable schemas for KairoRAG."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


SourceType = Literal["job", "resume", "company_doc", "interview_note"]


class VerificationStatus(str, Enum):
    """Lifecycle states for job records."""

    ACTIVE = "active"
    CLOSED = "closed"
    STALE = "stale"
    UNKNOWN = "unknown"
    UPDATED = "updated"
    DUPLICATE = "duplicate"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_content_hash(*parts: Any) -> str:
    """Create a stable short hash for job and chunk content."""

    payload = "\n".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


@dataclass
class JsonMixin:
    """Small serialization helper shared by dataclass schemas."""

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=_json_default, ensure_ascii=False))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class DocumentRecord(JsonMixin):
    doc_id: str
    source_type: SourceType
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord(JsonMixin):
    chunk_id: str
    doc_id: str
    source_type: SourceType
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class JobRecord(JsonMixin):
    job_id: str
    company: str
    title: str
    location: str
    jd_text: str
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    agent_related_keywords: list[str] = field(default_factory=list)
    job_url: str = ""
    source_platform: str = ""
    first_seen_at: str = ""
    last_verified_at: str = ""
    verification_status: str = VerificationStatus.UNKNOWN.value
    verification_confidence: float = 0.0
    evidence_urls: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    closed_reason: str = ""
    archived_at: str = ""
    content_hash: str = ""
    duplicate_of: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = stable_content_hash(
                self.company,
                self.title,
                self.location,
                self.jd_text,
                "|".join(self.required_skills),
                "|".join(self.preferred_skills),
            )

    @property
    def is_active_for_retrieval(self) -> bool:
        return self.verification_status not in {
            VerificationStatus.CLOSED.value,
            VerificationStatus.DUPLICATE.value,
        }


@dataclass
class SearchResult(JsonMixin):
    chunk_id: str
    score: float
    source_type: str
    title: str
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadChunkResult(JsonMixin):
    chunk_id: str
    text: str
    prev_chunk: str | None = None
    next_chunk: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation(JsonMixin):
    chunk_id: str
    source_type: str
    title: str
    evidence: str


@dataclass
class AgentState(JsonMixin):
    question: str
    search_results: list[SearchResult] = field(default_factory=list)
    read_chunks: list[ReadChunkResult] = field(default_factory=list)
    compressed_context: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer(JsonMixin):
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    verification_trace: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebSearchResult(JsonMixin):
    """Legacy / Deprecated：旧维护模块使用的 mock web search 结果。"""

    title: str
    url: str
    snippet: str
    source: str = "mock"
    score: float = 0.0


@dataclass
class JobPageEvidence(JsonMixin):
    url: str
    status_signal: str
    title_match: bool = False
    company_match: bool = False
    location_match: bool = False
    snippet: str = ""
    detected_phrases: list[str] = field(default_factory=list)
    retrieved_at: str = field(default_factory=utc_now)
    http_status: int | None = None
    error: str = ""


@dataclass
class JobVerificationResult(JsonMixin):
    job_id: str
    old_status: str
    new_status: str
    confidence: float
    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    updated_fields: dict[str, Any] = field(default_factory=dict)
    needs_reindex: bool = False
    verified_at: str = field(default_factory=utc_now)


@dataclass
class UpdateReport(JsonMixin):
    job_id: str
    action: str
    changed_fields: dict[str, Any]
    old_record: dict[str, Any]
    new_record: dict[str, Any]
    needs_reindex: bool
    audit_log_path: str


@dataclass
class IndexRefreshReport(JsonMixin):
    job_ids: list[str]
    mode: str
    rebuilt: bool
    index_dir: str
    refreshed_at: str = field(default_factory=utc_now)


@dataclass
class BudgetReport(JsonMixin):
    retrieved_chunks: list[str]
    read_chunks: list[str]
    estimated_context_tokens: int
    dropped_chunks: list[str]
