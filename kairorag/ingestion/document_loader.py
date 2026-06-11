"""High-level document loading for KairoRAG raw data."""

from __future__ import annotations

from pathlib import Path

from kairorag.ingestion.company_loader import load_company_docs, load_interview_notes
from kairorag.ingestion.jd_loader import jobs_to_documents, load_jobs
from kairorag.ingestion.resume_loader import load_resume
from kairorag.schemas import DocumentRecord, JobRecord


def load_knowledge_base(raw_dir: str | Path) -> tuple[list[DocumentRecord], list[JobRecord]]:
    """Load jobs, resume, company docs, and interview notes."""

    base = Path(raw_dir)
    jobs_path = base / "jobs.csv"
    jobs = load_jobs(jobs_path) if jobs_path.exists() else []
    docs = jobs_to_documents(jobs)

    resume = load_resume(base / "resume.md")
    if resume:
        docs.append(resume)

    docs.extend(load_company_docs(base / "company_docs"))
    docs.extend(load_interview_notes(base / "interview_notes"))
    return docs, jobs


def load_documents(raw_dir: str | Path) -> list[DocumentRecord]:
    """Load all retrieval documents."""

    docs, _jobs = load_knowledge_base(raw_dir)
    return docs

