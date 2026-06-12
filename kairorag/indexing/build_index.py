"""Deprecated：旧本地 pickle 索引构建入口已废弃。

该模块不再生成 `keyword_index.pkl` 或 `vector_store.pkl`。项目主路径已经迁移到
cloud-native 索引：OpenAI embedding、Qdrant 与 BM25 JSON 源文档。
"""

from __future__ import annotations

import sys
from pathlib import Path


DEPRECATED_MESSAGE = (
    "该入口已废弃。请使用 `kairo index` 或 `python -m kairorag.cloud.index`。"
)


def build_indexes(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, int]:
    """旧本地索引构建函数已禁用，调用时直接退出。"""

    _ = (input_dir, output_dir, chunk_size, overlap)
    print(DEPRECATED_MESSAGE)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    """打印废弃提示并以退出码 2 结束。"""

    _ = argv
    print(DEPRECATED_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
