"""Project configuration and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
MAINTENANCE_DIR = DATA_DIR / "maintenance"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class KairoConfig:
    """Runtime knobs with conservative local defaults."""

    raw_data_dir: Path = RAW_DATA_DIR
    index_dir: Path = INDEX_DIR
    eval_dir: Path = EVAL_DIR
    maintenance_dir: Path = MAINTENANCE_DIR
    chunk_size: int = 120
    chunk_overlap: int = 24
    embedding_dim: int = 128
    max_search_results: int = 10
    max_chunks_to_read: int = 5
    max_context_tokens: int = 1800
    max_tool_calls: int = 20
    freshness_threshold_days: int = 7


def ensure_project_dirs() -> None:
    """Create runtime directories used by CLIs."""

    for path in (RAW_DATA_DIR, INDEX_DIR, EVAL_DIR, MAINTENANCE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
