"""Apply verification results with soft-archive semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kairorag.config import RAW_DATA_DIR
from kairorag.ingestion.jd_loader import load_jobs, write_jobs
from kairorag.maintenance.audit_log import append_audit_log
from kairorag.schemas import JobRecord, JobVerificationResult, UpdateReport, VerificationStatus, stable_content_hash, utc_now


def _copy_job(job: JobRecord) -> JobRecord:
    return JobRecord(**job.to_dict())


def apply_verification_result(
    result: JobVerificationResult,
    mode: str = "soft",
    jobs_file: str | Path = RAW_DATA_DIR / "jobs.csv",
    apply: bool = False,
) -> UpdateReport:
    """Apply a verification result. Dry-run by default."""

    jobs = load_jobs(jobs_file)
    try:
        index = next(i for i, job in enumerate(jobs) if job.job_id == result.job_id)
    except StopIteration as exc:
        raise ValueError(f"job_id not found: {result.job_id}") from exc

    old = _copy_job(jobs[index])
    new = _copy_job(jobs[index])
    changed: dict[str, Any] = {}

    def set_field(name: str, value: Any) -> None:
        if getattr(new, name) != value:
            changed[name] = {"old": getattr(new, name), "new": value}
            setattr(new, name, value)

    status = result.new_status
    set_field("last_verified_at", result.verified_at)
    set_field("verification_status", status)
    set_field("verification_confidence", result.confidence)
    evidence_urls = [
        str(item.get("url"))
        for item in result.evidence
        if isinstance(item, dict) and item.get("url")
    ]
    evidence_snippets = [
        str(item.get("snippet"))
        for item in result.evidence
        if isinstance(item, dict) and item.get("snippet")
    ][:5]
    if evidence_urls:
        set_field("evidence_urls", evidence_urls)
    if evidence_snippets:
        set_field("evidence_snippets", evidence_snippets)

    if status == VerificationStatus.UPDATED.value:
        for key, value in result.updated_fields.items():
            if hasattr(new, key):
                set_field(key, value)
        set_field("verification_status", VerificationStatus.ACTIVE.value)
    elif status == VerificationStatus.CLOSED.value:
        set_field("closed_reason", result.reason)
        if mode == "soft":
            set_field("archived_at", result.verified_at)
    elif status == VerificationStatus.STALE.value:
        set_field("closed_reason", "unconfirmed stale evidence")
    elif status == VerificationStatus.DUPLICATE.value and result.updated_fields.get("duplicate_of"):
        set_field("duplicate_of", result.updated_fields["duplicate_of"])

    new.content_hash = stable_content_hash(
        new.company,
        new.title,
        new.location,
        new.jd_text,
        "|".join(new.required_skills),
        "|".join(new.preferred_skills),
    )
    if old.content_hash != new.content_hash:
        changed["content_hash"] = {"old": old.content_hash, "new": new.content_hash}

    audit_path = str(Path(jobs_file).parent.parent / "maintenance" / "audit_log.jsonl")
    if apply:
        jobs[index] = new
        write_jobs(jobs_file, jobs)
        audit_path = append_audit_log(
            {
                "job_id": result.job_id,
                "action": "archive" if status == VerificationStatus.CLOSED.value else "update",
                "old_status": old.verification_status,
                "new_status": new.verification_status,
                "confidence": result.confidence,
                "evidence_urls": evidence_urls,
                "reason": result.reason,
                "changed_fields": changed,
            },
            audit_path,
        )

    return UpdateReport(
        job_id=result.job_id,
        action="dry_run" if not apply else "update",
        changed_fields=changed,
        old_record=old.to_dict(),
        new_record=new.to_dict(),
        needs_reindex=result.needs_reindex or "content_hash" in changed,
        audit_log_path=audit_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a verification result JSON")
    parser.add_argument("--jobs-file", default=str(RAW_DATA_DIR / "jobs.csv"))
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--apply-updates", action="store_true")
    args = parser.parse_args()
    result = JobVerificationResult(**json.loads(args.result_json))
    report = apply_verification_result(result, jobs_file=args.jobs_file, apply=args.apply_updates)
    print(report.to_json())


if __name__ == "__main__":
    main()

