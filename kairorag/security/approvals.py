"""破坏性写回操作的审批决策器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from kairorag.security.policies import ApprovalPolicy, DestructiveAction, VALID_APPROVAL_POLICIES


@dataclass(frozen=True)
class ApprovalRequest:
    """描述一次需要审批的破坏性操作。"""

    action: DestructiveAction
    resource_id: str | None
    summary: str
    risk_level: str
    metadata_patch: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ApprovalDecision:
    """审批结果，调用方必须按 dry_run 和 allowed 执行。"""

    allowed: bool
    dry_run: bool
    reason: str
    requires_confirmation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ApprovalManager:
    """根据统一策略决定是否允许真实写回。"""

    def __init__(self, policy: ApprovalPolicy = "require_confirmation") -> None:
        normalized = str(policy or "require_confirmation").strip().lower()
        if normalized not in VALID_APPROVAL_POLICIES:
            raise ValueError(f"不支持的审批策略：{policy}")
        self.policy: ApprovalPolicy = normalized  # type: ignore[assignment]

    def decide(
        self,
        request: ApprovalRequest,
        *,
        user_confirmed: bool = False,
        agent_initiated: bool = False,
    ) -> ApprovalDecision:
        """返回审批决策；agent 发起的写回永远不能绕过 dry-run。"""

        if self.policy == "deny":
            return ApprovalDecision(
                allowed=False,
                dry_run=True,
                reason=f"审批策略 deny 已拒绝操作：{request.action}",
                requires_confirmation=False,
            )
        if self.policy == "dry_run":
            return ApprovalDecision(
                allowed=True,
                dry_run=True,
                reason=f"审批策略 dry_run 仅允许演练：{request.action}",
                requires_confirmation=False,
            )
        if agent_initiated:
            return ApprovalDecision(
                allowed=True,
                dry_run=True,
                reason="Agent 发起的破坏性操作必须保持 dry-run，不能自主真实写回。",
                requires_confirmation=True,
            )
        if self.policy == "require_confirmation":
            if user_confirmed:
                return ApprovalDecision(
                    allowed=True,
                    dry_run=False,
                    reason="用户已显式确认，允许真实写回。",
                    requires_confirmation=False,
                )
            return ApprovalDecision(
                allowed=True,
                dry_run=True,
                reason="需要用户显式确认；当前仅允许 dry-run。",
                requires_confirmation=True,
            )
        if user_confirmed:
            return ApprovalDecision(
                allowed=True,
                dry_run=False,
                reason="审批策略 allow 且用户已确认，允许真实写回。",
                requires_confirmation=False,
            )
        return ApprovalDecision(
            allowed=True,
            dry_run=True,
            reason="审批策略 allow 仍需要 CLI --yes 或人工确认后才真实写回。",
            requires_confirmation=True,
        )
