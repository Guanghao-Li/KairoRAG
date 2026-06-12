"""KairoRAG 配置入口。

本模块继续导出历史本地 demo 使用的路径常量，同时暴露新的 cloud-native
配置对象，避免旧导入路径在阶段一重构中失效。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kairorag.config.settings import (
    CloudRuntimeConfigurationError,
    KairoCloudSettings,
    validate_cloud_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
MAINTENANCE_DIR = DATA_DIR / "maintenance"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class KairoConfig:
    """Legacy / Deprecated：离线 demo 的本地路径和预算默认值。"""

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
    """创建 CLI 运行时仍会使用的本地目录。"""

    for path in (RAW_DATA_DIR, INDEX_DIR, EVAL_DIR, MAINTENANCE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CloudRuntimeConfigurationError",
    "DATA_DIR",
    "EVAL_DIR",
    "INDEX_DIR",
    "KairoCloudSettings",
    "KairoConfig",
    "MAINTENANCE_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "ensure_project_dirs",
    "validate_cloud_runtime",
]
