"""破坏性操作的审批策略定义。"""

from __future__ import annotations

from typing import Literal


ApprovalPolicy = Literal["deny", "dry_run", "require_confirmation", "allow"]

DestructiveAction = Literal[
    "mark_closed",
    "archive",
    "update_url",
    "mark_duplicate",
    "apply_freshness_update",
    "qdrant_payload_update",
]

VALID_APPROVAL_POLICIES: set[str] = {"deny", "dry_run", "require_confirmation", "allow"}
