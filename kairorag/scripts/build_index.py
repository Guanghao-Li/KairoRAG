"""Deprecated：旧脚本路径已废弃，请使用 cloud-native 索引入口。"""

from __future__ import annotations

import sys

from kairorag.indexing.build_index import DEPRECATED_MESSAGE


def main(argv: list[str] | None = None) -> int:
    """打印废弃提示并以退出码 2 结束。"""

    _ = argv
    print(DEPRECATED_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
