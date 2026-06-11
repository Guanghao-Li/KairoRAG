"""Evaluation report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(path: str | Path, payload: dict[str, Any], title: str = "KairoRAG 评测报告") -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## 总体指标", ""]
    metrics = payload.get("overall_metrics", {})
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | ---: |")
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.3f} |" if isinstance(value, (int, float)) else f"| {key} | {value} |")
    lines.extend(["", "## 逐条样例", ""])
    for case in payload.get("cases", []):
        lines.append(f"### {case.get('id', case.get('job_id', 'case'))}")
        if case.get("question"):
            lines.append(f"- 问题：{case['question']}")
        if case.get("gold_chunk_ids") is not None:
            read_chunks = case.get("read_chunks", [])
            gold = case.get("gold_chunk_ids", [])
            hit = sorted(set(gold) & set(read_chunks))
            miss = sorted(set(gold) - set(read_chunks))
            lines.append(f"- 命中的 gold chunks：{hit}")
            lines.append(f"- 未命中的 gold chunks：{miss}")
        if case.get("retrieval_path"):
            lines.append(f"- 检索路径：{' -> '.join(case['retrieval_path'])}")
        if case.get("prediction"):
            lines.append(f"- 预测结果：`{case['prediction']}`")
        if case.get("metrics"):
            lines.append(f"- 指标明细：`{json.dumps(case['metrics'], ensure_ascii=False)}`")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
