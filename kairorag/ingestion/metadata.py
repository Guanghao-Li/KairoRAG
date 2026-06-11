"""Metadata helpers for loaders and index filters."""

from __future__ import annotations

import re
from typing import Any

from kairorag.schemas import VerificationStatus


def parse_list(value: Any) -> list[str]:
    """Parse pipe/semicolon/comma separated cells into a clean list."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[|;,]", text)
    return [part.strip() for part in parts if part.strip()]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate that works for English and CJK text."""

    if not text:
        return 0
    ascii_words = re.findall(r"[A-Za-z0-9_#+.-]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    other = max(0, len(text) - sum(len(w) for w in ascii_words) - len(cjk_chars))
    return max(1, len(ascii_words) + len(cjk_chars) + other // 4)


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing user-visible content too much."""

    return re.sub(r"\s+", " ", text or "").strip()


def normalize_id_part(text: str) -> str:
    """Create stable lowercase id fragments."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "doc"


def active_for_retrieval(status: str | None, include_archived: bool = False) -> bool:
    """Return whether a record should be visible in default retrieval."""

    if include_archived:
        return True
    return status not in {VerificationStatus.CLOSED.value, VerificationStatus.DUPLICATE.value}


def metadata_matches(
    metadata: dict[str, Any],
    filters: dict[str, Any] | None = None,
    include_archived: bool = False,
) -> bool:
    """Apply exact-match metadata filters and default inactive exclusions."""

    status = str(metadata.get("verification_status", "")).lower()
    if not active_for_retrieval(status, include_archived=include_archived):
        return False
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "include_archived":
            continue
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

