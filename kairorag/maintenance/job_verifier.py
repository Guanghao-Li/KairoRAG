"""Evidence-based job freshness verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kairorag.config import RAW_DATA_DIR
from kairorag.ingestion.jd_loader import load_jobs
from kairorag.maintenance.job_page_parser import parse_job_page
from kairorag.maintenance.knowledge_base_updater import apply_verification_result
from kairorag.maintenance.web_search import search_web
from kairorag.schemas import JobRecord, JobVerificationResult, VerificationStatus


def verify_job(job: JobRecord, search: bool = True) -> JobVerificationResult:
    """Verify a job from original URL and mockable web evidence."""

    if job.verification_status == VerificationStatus.DUPLICATE.value:
        return JobVerificationResult(
            job_id=job.job_id,
            old_status=job.verification_status,
            new_status=VerificationStatus.DUPLICATE.value,
            confidence=max(job.verification_confidence, 0.85),
            reason=f"Record is marked duplicate_of={job.duplicate_of}.",
            evidence=[{"type": "metadata", "duplicate_of": job.duplicate_of, "url": job.job_url}],
            updated_fields={"duplicate_of": job.duplicate_of},
            needs_reindex=True,
        )

    evidence = []
    original = None
    if job.job_url:
        original = parse_job_page(
            job.job_url,
            expected_title=job.title,
            expected_company=job.company,
            expected_location=job.location,
        )
        evidence.append(original.to_dict())
        if original.status_signal == "active":
            return JobVerificationResult(
                job_id=job.job_id,
                old_status=job.verification_status,
                new_status=VerificationStatus.ACTIVE.value,
                confidence=0.94 if original.company_match or original.title_match else 0.82,
                reason="Original job URL has an active application signal.",
                evidence=evidence,
                updated_fields={},
                needs_reindex=False,
            )
        if original.status_signal == "closed":
            return JobVerificationResult(
                job_id=job.job_id,
                old_status=job.verification_status,
                new_status=VerificationStatus.CLOSED.value,
                confidence=0.93,
                reason="Original job URL shows a closed or no-longer-available signal.",
                evidence=evidence,
                updated_fields={},
                needs_reindex=job.verification_status != VerificationStatus.CLOSED.value,
            )

    if search:
        query = f"{job.company} {job.title} {job.location} {job.job_url} {job.source_platform}"
        results = search_web(query, top_k=5)
        evidence.extend({"type": "web_search", **result.to_dict()} for result in results)
        for result in results:
            parsed = parse_job_page(
                result.url,
                expected_title=job.title,
                expected_company=job.company,
                expected_location=job.location,
            )
            evidence.append(parsed.to_dict())
            if parsed.status_signal == "active":
                status = VerificationStatus.UPDATED.value if result.url != job.job_url else VerificationStatus.ACTIVE.value
                return JobVerificationResult(
                    job_id=job.job_id,
                    old_status=job.verification_status,
                    new_status=status,
                    confidence=0.9 if parsed.company_match or parsed.title_match else 0.72,
                    reason="Web search found an active posting with page evidence.",
                    evidence=evidence,
                    updated_fields={"job_url": result.url} if result.url != job.job_url else {},
                    needs_reindex=status == VerificationStatus.UPDATED.value,
                )

    return JobVerificationResult(
        job_id=job.job_id,
        old_status=job.verification_status,
        new_status=VerificationStatus.STALE.value,
        confidence=0.25,
        reason="Could not confirm active or closed status from URL/page/snippet evidence.",
        evidence=evidence,
        updated_fields={},
        needs_reindex=job.verification_status != VerificationStatus.STALE.value,
    )


def _select_jobs(jobs: list[JobRecord], job_id: str = "", statuses: str = "", limit: int | None = None) -> list[JobRecord]:
    selected = jobs
    if job_id:
        selected = [job for job in selected if job.job_id == job_id]
    if statuses:
        wanted = {item.strip() for item in statuses.split(",") if item.strip()}
        selected = [job for job in selected if job.verification_status in wanted]
    return selected[:limit] if limit else selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify job freshness")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--jobs-file", default=str(RAW_DATA_DIR / "jobs.csv"))
    parser.add_argument("--status", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--search-web", action="store_true")
    parser.add_argument("--apply-updates", action="store_true")
    args = parser.parse_args()

    jobs = load_jobs(args.jobs_file)
    selected = _select_jobs(jobs, args.job_id, args.status, args.limit)
    results = []
    for job in selected:
        result = verify_job(job, search=args.search_web or True)
        results.append(result.to_dict())
        if args.apply_updates:
            apply_verification_result(result, jobs_file=args.jobs_file, apply=True)
    print(json.dumps(results if len(results) != 1 else results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
