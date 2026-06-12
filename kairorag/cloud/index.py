"""Cloud indexing CLI。"""

from __future__ import annotations

import argparse
import logging

from kairorag.cloud.indexing import CloudIndexer
from kairorag.cloud.runtime import build_cloud_runtime
from kairorag.config import KairoCloudSettings
from kairorag.config.settings import validate_cloud_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 KairoRAG cloud 索引。")
    parser.add_argument("--recreate", action="store_true", help="重建 Qdrant collection。")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 CLOUD_CHUNK_BATCH_SIZE。")
    parser.add_argument("--manifest-path", default=None, help="覆盖 cloud index manifest 输出路径。")
    parser.add_argument("--bm25-path", default=None, help="覆盖 cloud BM25 JSON 输出路径。")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    settings = validate_cloud_runtime(KairoCloudSettings())
    if args.batch_size is not None:
        if args.batch_size <= 0:
            print("错误：--batch-size 必须大于 0。")
            return 2
        settings.cloud_chunk_batch_size = args.batch_size
    if args.manifest_path:
        settings.cloud_index_manifest_path = args.manifest_path
    if args.bm25_path:
        settings.cloud_bm25_index_path = args.bm25_path
    runtime = build_cloud_runtime(settings)
    indexer = CloudIndexer(
        settings=runtime.settings,
        embedding_provider=runtime.embedding_provider,
        vector_store=runtime.vector_store,
        keyword_search=runtime.keyword_search,
    )
    stats = indexer.build(recreate=args.recreate)
    print("Cloud 索引构建完成")
    print(f"加载文档数：{stats.documents_loaded}")
    print(f"chunk 数：{stats.chunks_created}")
    print(f"embedding 数：{stats.embeddings_created}")
    print(f"Qdrant upsert 数：{stats.qdrant_upserted}")
    print(f"BM25 index 数：{stats.bm25_documents_indexed}")
    print(f"manifest 路径：{stats.manifest_path}")
    print(f"BM25 路径：{stats.bm25_index_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
