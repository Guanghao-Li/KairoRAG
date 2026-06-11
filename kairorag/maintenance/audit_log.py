"""Append-only audit log for knowledge-base updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kairorag.config import MAINTENANCE_DIR
from kairorag.schemas import utc_now


def append_audit_log(
    entry: dict[str, Any],
    path: str | Path = MAINTENANCE_DIR / "audit_log.jsonl",
) -> str:
    """Append one audit entry and return the log path."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": utc_now(), **entry}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(log_path)

