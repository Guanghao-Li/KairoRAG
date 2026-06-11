"""Company and interview-note loaders."""

from __future__ import annotations

from pathlib import Path

from kairorag.ingestion.metadata import normalize_id_part
from kairorag.schemas import DocumentRecord


def load_markdown_directory(path: str | Path, source_type: str, prefix: str) -> list[DocumentRecord]:
    """Load all markdown/text files in a directory as documents."""

    directory = Path(path)
    if not directory.exists():
        return []
    docs: list[DocumentRecord] = []
    for file_path in sorted(directory.glob("*")):
        if file_path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            continue
        stem = normalize_id_part(file_path.stem)
        docs.append(
            DocumentRecord(
                doc_id=f"{prefix}_{stem}",
                source_type=source_type,  # type: ignore[arg-type]
                title=file_path.stem.replace("_", " ").title(),
                text=file_path.read_text(encoding="utf-8"),
                metadata={
                    "path": str(file_path),
                    "verification_status": "active",
                    "source_file": file_path.name,
                },
            )
        )
    return docs


def load_company_docs(path: str | Path) -> list[DocumentRecord]:
    return load_markdown_directory(path, "company_doc", "company")


def load_interview_notes(path: str | Path) -> list[DocumentRecord]:
    return load_markdown_directory(path, "interview_note", "interview")

