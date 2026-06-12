"""岗位 freshness verification 的结构化 schema。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


JobFreshnessStatus = Literal["active", "closed", "stale", "updated", "duplicate", "unknown"]
JobFreshnessAction = Literal[
    "keep_active",
    "mark_closed",
    "mark_stale",
    "update_url",
    "mark_duplicate",
    "manual_review",
]

VALID_STATUSES = {"active", "closed", "stale", "updated", "duplicate", "unknown"}
VALID_ACTIONS = {"keep_active", "mark_closed", "mark_stale", "update_url", "mark_duplicate", "manual_review"}


@dataclass(frozen=True)
class JobFreshnessEvidence:
    source: str
    url: str
    title: str
    snippet: str
    signal: str
    confidence: float
    observed_at: str

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence, "evidence.confidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobFreshnessResult:
    job_id: str | None
    chunk_id: str | None
    company: str | None
    title: str | None
    original_url: str | None
    status: JobFreshnessStatus
    confidence: float
    evidence: list[JobFreshnessEvidence]
    recommended_action: str
    reason: str
    checked_at: str
    new_url: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"不支持的 freshness status：{self.status}")
        if self.recommended_action not in VALID_ACTIONS:
            raise ValueError(f"不支持的 freshness recommended_action：{self.recommended_action}")
        _validate_confidence(self.confidence, "confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "chunk_id": self.chunk_id,
            "company": self.company,
            "title": self.title,
            "original_url": self.original_url,
            "status": self.status,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "recommended_action": self.recommended_action,
            "reason": self.reason,
            "checked_at": self.checked_at,
            "new_url": self.new_url,
        }


def _validate_confidence(value: float, field_name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{field_name} 必须在 0 到 1 之间。")
