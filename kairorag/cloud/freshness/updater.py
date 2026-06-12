"""岗位 freshness metadata 的安全写回计划与可选执行。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.observability import TraceEvent, TraceLogger, now_iso, redact_secrets, safe_json_dumps
from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.vectorstores import VectorStoreProvider


@dataclass(frozen=True)
class FreshnessUpdatePlan:
    should_update: bool
    action: str
    chunk_id: str | None
    metadata_patch: dict[str, Any]
    requires_reindex: bool
    reason: str
    dry_run: bool
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class FreshnessUpdateResult:
    applied: bool
    plan: FreshnessUpdatePlan
    audit_log_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class CloudFreshnessUpdater:
    """生成 freshness metadata patch，并在用户授权时安全写回 Qdrant。"""

    def __init__(self, settings: KairoCloudSettings, trace_logger: TraceLogger | None = None) -> None:
        self.settings = settings
        self.trace_logger = trace_logger or TraceLogger(
            settings.trace_log_path,
            enabled=settings.observability_enabled,
        )

    def plan_update(self, result: JobFreshnessResult) -> FreshnessUpdatePlan:
        return self._plan_update(result, dry_run=self.settings.qdrant_metadata_write_dry_run)

    def apply_update(
        self,
        result: JobFreshnessResult,
        *,
        vector_store: VectorStoreProvider,
        dry_run: bool = True,
    ) -> FreshnessUpdateResult:
        plan = self._plan_update(result, dry_run=dry_run)
        self._log_trace("freshness_update_planned", result, {"plan": plan.to_dict()})
        audit_path = self._write_audit(result, plan, status="planned")

        if not plan.should_update:
            return FreshnessUpdateResult(applied=False, plan=plan, audit_log_path=audit_path)
        if dry_run:
            return FreshnessUpdateResult(applied=False, plan=plan, audit_log_path=audit_path)
        if not self.settings.qdrant_metadata_write_enabled:
            error = "Qdrant metadata 写回未启用：请设置 QDRANT_METADATA_WRITE_ENABLED=true 后再 apply。"
            self._write_audit(result, plan, status="error", error=error)
            self._log_trace("provider_error", result, {"error": error})
            return FreshnessUpdateResult(applied=False, plan=plan, audit_log_path=audit_path, error=error)
        if not plan.chunk_id:
            error = "无法写回 freshness metadata：缺少 chunk_id。"
            self._write_audit(result, plan, status="error", error=error)
            self._log_trace("provider_error", result, {"error": error})
            return FreshnessUpdateResult(applied=False, plan=plan, audit_log_path=audit_path, error=error)

        try:
            vector_store.update_payload(
                [plan.chunk_id],
                plan.metadata_patch,
                allowed_fields=self.settings.qdrant_metadata_allowed_fields,
            )
        except Exception as exc:
            error = str(exc) if isinstance(exc, KairoProviderError) else "Qdrant metadata 写回失败。"
            self._write_audit(result, plan, status="error", error=error)
            self._log_trace("provider_error", result, {"error": error})
            return FreshnessUpdateResult(applied=False, plan=plan, audit_log_path=audit_path, error=error)

        self._write_audit(result, plan, status="applied")
        self._log_trace("freshness_update_applied", result, {"plan": plan.to_dict(), "applied": True})
        return FreshnessUpdateResult(applied=True, plan=plan, audit_log_path=audit_path)

    def _plan_update(self, result: JobFreshnessResult, *, dry_run: bool) -> FreshnessUpdatePlan:
        base_patch = {
            "verification_status": result.status,
            "last_verified_at": result.checked_at,
            "verification_confidence": result.confidence,
        }
        if result.status == "active":
            return _plan(
                result,
                should_update=True,
                action="keep_active",
                metadata_patch={**base_patch, "archived": False, "manual_review_required": False},
                requires_reindex=False,
                dry_run=dry_run,
            )
        if result.status == "closed":
            return _plan(
                result,
                should_update=True,
                action="mark_closed",
                metadata_patch={
                    **base_patch,
                    "archived": True,
                    "archived_at": result.checked_at,
                    "closed_reason": result.reason,
                    "manual_review_required": False,
                },
                requires_reindex=True,
                dry_run=dry_run,
            )
        if result.status == "updated":
            patch = {**base_patch, "archived": False, "manual_review_required": False}
            if result.new_url:
                patch["original_url"] = result.new_url
                patch["canonical_url"] = result.new_url
            return _plan(
                result,
                should_update=True,
                action="update_url",
                metadata_patch=patch,
                requires_reindex=True,
                dry_run=dry_run,
            )
        if result.status == "duplicate":
            patch = {
                **base_patch,
                "archived": True,
                "archived_at": result.checked_at,
                "manual_review_required": False,
            }
            if result.new_url:
                patch["duplicate_of"] = result.new_url
            return _plan(
                result,
                should_update=True,
                action="mark_duplicate",
                metadata_patch=patch,
                requires_reindex=True,
                dry_run=dry_run,
            )
        if result.status == "stale":
            return _plan(
                result,
                should_update=True,
                action="mark_stale",
                metadata_patch={**base_patch, "archived": False, "manual_review_required": True},
                requires_reindex=False,
                dry_run=dry_run,
            )
        return _plan(
            result,
            should_update=True,
            action="manual_review",
            metadata_patch={**base_patch, "archived": False, "manual_review_required": True},
            requires_reindex=False,
            dry_run=dry_run,
        )

    def _write_audit(
        self,
        result: JobFreshnessResult,
        plan: FreshnessUpdatePlan,
        *,
        status: str,
        error: str | None = None,
    ) -> str:
        path = Path(self.settings.freshness_audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": now_iso(),
            "action": plan.action,
            "chunk_id": plan.chunk_id,
            "status": status,
            "confidence": result.confidence,
            "dry_run": plan.dry_run,
            "metadata_patch": plan.metadata_patch,
            "evidence_count": len(result.evidence),
            "reason": result.reason,
        }
        if error:
            payload["error"] = error
        with path.open("a", encoding="utf-8") as handle:
            handle.write(safe_json_dumps(redact_secrets(payload)) + "\n")
        return str(path)

    def _log_trace(self, event_type: str, result: JobFreshnessResult, detail: dict[str, Any]) -> None:
        self.trace_logger.log(
            TraceEvent(
                event_type=event_type,
                timestamp=now_iso(),
                run_id=f"freshness-{uuid4()}",
                query=None,
                detail={
                    "chunk_id": result.chunk_id,
                    "job_id": result.job_id,
                    "status": result.status,
                    **detail,
                },
            )
        )


def _plan(
    result: JobFreshnessResult,
    *,
    should_update: bool,
    action: str,
    metadata_patch: dict[str, Any],
    requires_reindex: bool,
    dry_run: bool,
) -> FreshnessUpdatePlan:
    return FreshnessUpdatePlan(
        should_update=should_update,
        action=action,
        chunk_id=result.chunk_id,
        job_id=result.job_id,
        metadata_patch=metadata_patch,
        requires_reindex=requires_reindex,
        reason=result.reason,
        dry_run=dry_run,
    )
