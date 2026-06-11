"""Freshness checks for dynamic job verification."""

from __future__ import annotations

from datetime import datetime, timezone

from kairorag.schemas import VerificationStatus


def needs_verification(
    last_verified_at: str,
    status: str,
    threshold_days: int = 7,
    explicit: bool = False,
) -> bool:
    """Return whether a job should be re-verified."""

    if explicit:
        return True
    if status in {VerificationStatus.UNKNOWN.value, VerificationStatus.STALE.value}:
        return True
    if not last_verified_at:
        return True
    try:
        parsed = datetime.fromisoformat(last_verified_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    age_days = (datetime.now(timezone.utc) - parsed).days
    return age_days > threshold_days

