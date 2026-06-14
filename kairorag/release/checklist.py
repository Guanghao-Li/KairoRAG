"""发布检查清单。"""

from __future__ import annotations


RELEASE_CHECKLIST_ITEMS: list[str] = [
    "运行 python -m pytest。",
    "运行 python -m ruff check .。",
    "执行 secret scan，确认没有真实 API key。",
    "运行 kairo doctor --repo。",
    "运行 kairo doctor --release。",
    "运行 kairo eval --suite all --fake-providers。",
    "执行 docker build 或 docker compose smoke test。",
    "确认 README 已更新。",
    "确认 legacy 主路径没有恢复。",
    "确认 live tests 默认跳过。",
    "确认 .env 没有提交。",
    "确认 trace / audit / eval output 没有提交。",
    "确认版本号和发布说明正确。",
]


def build_release_checklist() -> str:
    """生成 Markdown 形式的发布检查清单。"""

    lines = ["# KairoRAG 发布检查清单", ""]
    for item in RELEASE_CHECKLIST_ITEMS:
        lines.append(f"- [ ] {item}")
    return "\n".join(lines) + "\n"
