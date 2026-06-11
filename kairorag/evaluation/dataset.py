"""Evaluation dataset readers."""

from __future__ import annotations

import json
from pathlib import Path

from kairorag.config import EVAL_DIR


def load_qa_eval(path: str | Path = EVAL_DIR / "qa_eval.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_job_verification_eval(path: str | Path = EVAL_DIR / "job_verification_eval.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

