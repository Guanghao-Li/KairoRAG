"""Detect and soft-mark duplicate jobs."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from kairorag.config import RAW_DATA_DIR
from kairorag.ingestion.jd_loader import load_jobs, write_jobs
from kairorag.maintenance.audit_log import append_audit_log
from kairorag.schemas import VerificationStatus, utc_now


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def detect_duplicate_jobs(jobs_file: str | Path, threshold: float = 0.85, apply: bool = False) -> list[dict[str, object]]:
    """Detect likely duplicate jobs and optionally mark duplicates."""

    jobs = load_jobs(jobs_file)
    findings: list[dict[str, object]] = []
    for i, left in enumerate(jobs):
        for right in jobs[i + 1 :]:
            score = 0.0
            if left.job_url and left.job_url == right.job_url:
                score = 1.0
            elif left.content_hash and left.content_hash == right.content_hash:
                score = 1.0
            else:
                title_score = _similarity(left.title, right.title)
                company_score = _similarity(left.company, right.company)
                location_score = _similarity(left.location, right.location)
                skill_overlap = len(set(left.required_skills) & set(right.required_skills)) / max(
                    1, len(set(left.required_skills) | set(right.required_skills))
                )
                score = 0.35 * title_score + 0.25 * company_score + 0.15 * location_score + 0.25 * skill_overlap
            if score >= threshold:
                primary = left if left.priority <= right.priority else right
                duplicate = right if primary is left else left
                findings.append(
                    {
                        "primary_job_id": primary.job_id,
                        "duplicate_job_id": duplicate.job_id,
                        "score": round(score, 3),
                    }
                )
                if apply:
                    duplicate.verification_status = VerificationStatus.DUPLICATE.value
                    duplicate.duplicate_of = primary.job_id
                    duplicate.last_verified_at = utc_now()
                    append_audit_log(
                        {
                            "job_id": duplicate.job_id,
                            "action": "mark_duplicate",
                            "old_status": "unknown",
                            "new_status": "duplicate",
                            "confidence": score,
                            "evidence_urls": [duplicate.job_url],
                            "reason": f"Duplicate of {primary.job_id}",
                            "changed_fields": {"duplicate_of": primary.job_id},
                        },
                        Path(jobs_file).parent.parent / "maintenance" / "audit_log.jsonl",
                    )
    if apply:
        write_jobs(jobs_file, jobs)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate job records")
    parser.add_argument("--jobs-file", default=str(RAW_DATA_DIR / "jobs.csv"))
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--apply-updates", action="store_true")
    args = parser.parse_args()
    findings = detect_duplicate_jobs(args.jobs_file, threshold=args.threshold, apply=args.apply_updates)
    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

