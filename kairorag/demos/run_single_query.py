"""Deprecated：旧离线单次查询入口已废弃。

该模块仅保留给历史路径兼容。它不会再创建 legacy ReactAgent，也不会执行旧
RAG 流程。请使用统一 cloud-native CLI。
"""

from __future__ import annotations

import sys


DEPRECATED_MESSAGE = "该入口已废弃。请使用 `kairo query` 或 `python -m kairorag.cloud.query_cli`。"


def main(argv: list[str] | None = None) -> int:
    """打印废弃提示并以退出码 2 结束。"""

    _ = argv
    print(DEPRECATED_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
