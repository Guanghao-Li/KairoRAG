"""Load job descriptions from CSV into typed records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from kairorag.ingestion.metadata import parse_list
from kairorag.schemas import DocumentRecord, JobRecord, stable_content_hash


def _row_to_job(row: dict[str, str]) -> JobRecord:
    required = parse_list(row.get("required_skills"))
    preferred = parse_list(row.get("preferred_skills"))
    keywords = parse_list(row.get("agent_related_keywords"))
    evidence_urls = parse_list(row.get("evidence_urls"))
    evidence_snippets = parse_list(row.get("evidence_snippets"))
    content_hash = row.get("content_hash") or stable_content_hash(
        row.get("company", ""),
        row.get("title", ""),
        row.get("location", ""),
        row.get("jd_text", ""),
        "|".join(required),
        "|".join(preferred),
    )
    return JobRecord(
        job_id=row.get("job_id", "").strip(),
        company=row.get("company", "").strip(),
        title=row.get("title", "").strip(),
        location=row.get("location", "").strip(),
        jd_text=row.get("jd_text", "").strip(),
        required_skills=required,
        preferred_skills=preferred,
        agent_related_keywords=keywords,
        job_url=row.get("job_url", "").strip(),
        source_platform=row.get("source_platform", "").strip(),
        first_seen_at=row.get("first_seen_at", "").strip(),
        last_verified_at=row.get("last_verified_at", "").strip(),
        verification_status=(row.get("verification_status") or "unknown").strip(),
        verification_confidence=float(row.get("verification_confidence") or 0.0),
        evidence_urls=evidence_urls,
        evidence_snippets=evidence_snippets,
        closed_reason=row.get("closed_reason", "").strip(),
        archived_at=row.get("archived_at", "").strip(),
        content_hash=content_hash,
        duplicate_of=row.get("duplicate_of", "").strip(),
        priority=int(row.get("priority") or 0),
    )


def load_jobs(path: str | Path) -> list[JobRecord]:
    """Load job records from `jobs.csv`."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [_row_to_job(row) for row in csv.DictReader(handle)]


def jobs_to_documents(jobs: Iterable[JobRecord]) -> list[DocumentRecord]:
    """Convert job records into retrieval documents."""

    docs: list[DocumentRecord] = []
    for job in jobs:
        skills = ", ".join(job.required_skills + job.preferred_skills)
        text = (
            f"Company: {job.company}\n"
            f"Title: {job.title}\n"
            f"Location: {job.location}\n"
            f"Status: {job.verification_status}\n"
            f"Required skills: {', '.join(job.required_skills)}\n"
            f"Preferred skills: {', '.join(job.preferred_skills)}\n"
            f"Agent keywords: {', '.join(job.agent_related_keywords)}\n"
            f"Job URL: {job.job_url}\n\n"
            f"{job.jd_text}\n\n"
            f"Skill summary: {skills}"
        )
        docs.append(
            DocumentRecord(
                doc_id=job.job_id,
                source_type="job",
                title=f"{job.company} - {job.title}",
                text=text,
                metadata={
                    **job.to_dict(),
                    "active_for_retrieval": job.is_active_for_retrieval,
                },
            )
        )
    return docs


def write_jobs(path: str | Path, jobs: list[JobRecord]) -> None:
    """Write job records back to CSV without dropping unknown columns."""

    fieldnames = [
        "job_id",
        "company",
        "title",
        "location",
        "jd_text",
        "required_skills",
        "preferred_skills",
        "agent_related_keywords",
        "job_url",
        "source_platform",
        "first_seen_at",
        "last_verified_at",
        "verification_status",
        "verification_confidence",
        "evidence_urls",
        "evidence_snippets",
        "closed_reason",
        "archived_at",
        "content_hash",
        "duplicate_of",
        "priority",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            row = job.to_dict()
            for key in ("required_skills", "preferred_skills", "agent_related_keywords", "evidence_urls", "evidence_snippets"):
                row[key] = "|".join(row.get(key) or [])
            writer.writerow({name: row.get(name, "") for name in fieldnames})

