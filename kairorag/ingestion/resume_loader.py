"""Resume loader."""

from __future__ import annotations

from pathlib import Path

from kairorag.schemas import DocumentRecord


def load_resume(path: str | Path) -> DocumentRecord | None:
    """Load a markdown resume if present."""

    resume_path = Path(path)
    if not resume_path.exists():
        return None
    return DocumentRecord(
        doc_id="resume",
        source_type="resume",
        title="Candidate Resume",
        text=resume_path.read_text(encoding="utf-8"),
        metadata={"path": str(resume_path), "verification_status": "active"},
    )

