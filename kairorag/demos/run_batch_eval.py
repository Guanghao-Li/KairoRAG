"""Deprecated：旧离线批量评测入口已废弃。

请使用 `kairo eval` 生成 cloud-native 离线评测报告和 dashboard。
"""

from __future__ import annotations

import sys


DEPRECATED_MESSAGE = "该入口已废弃。请使用 `kairo eval --suite all --fake-providers`。"


def main(argv: list[str] | None = None) -> int:
    """打印废弃提示并以退出码 2 结束。"""

    _ = argv
    print(DEPRECATED_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
