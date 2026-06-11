"""Refresh affected indexes, falling back to rebuild."""

from __future__ import annotations

import argparse

from kairorag.config import INDEX_DIR, RAW_DATA_DIR
from kairorag.indexing.build_index import build_indexes
from kairorag.schemas import IndexRefreshReport


def refresh_index_for_jobs(job_ids: list[str]) -> IndexRefreshReport:
    """Fallback rebuild for affected job IDs."""

    build_indexes(RAW_DATA_DIR, INDEX_DIR)
    return IndexRefreshReport(job_ids=job_ids, mode="fallback_rebuild", rebuilt=True, index_dir=str(INDEX_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh indexes for affected jobs")
    parser.add_argument("--job-ids", required=True)
    args = parser.parse_args()
    report = refresh_index_for_jobs([item.strip() for item in args.job_ids.split(",") if item.strip()])
    print(report.to_json())


if __name__ == "__main__":
    main()

