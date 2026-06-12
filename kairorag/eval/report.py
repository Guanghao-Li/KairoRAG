"""Eval JSON 与 Markdown 报告生成。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_json_report(result: Any, path: str | Path) -> Path:
    """写出完整 JSON eval 报告。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_markdown_report(result: Any, path: str | Path) -> Path:
    """写出包含 suite 表格、指标和失败摘要的 Markdown 报告。"""

    payload = _to_jsonable(result)
    suites = payload.get("suites", [payload] if payload.get("suite") else [])
    lines = ["# KairoRAG Eval 报告", ""]
    lines.append(f"- suite 数量：{len(suites)}")
    lines.append(f"- 总 case 数：{sum(int(item.get('case_count', 0)) for item in suites)}")
    lines.append(f"- 通过 case 数：{sum(int(item.get('passed_count', 0)) for item in suites)}")
    lines.append(f"- 失败 case 数：{sum(int(item.get('failed_count', 0)) for item in suites)}")
    lines.append("")
    lines.append("| Suite | Case | Passed | Failed |")
    lines.append("| --- | ---: | ---: | ---: |")
    for suite in suites:
        lines.append(
            f"| {suite.get('suite')} | {suite.get('case_count', 0)} | "
            f"{suite.get('passed_count', 0)} | {suite.get('failed_count', 0)} |"
        )
    lines.append("")
    for suite in suites:
        lines.append(f"## {suite.get('suite')}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, value in sorted((suite.get("aggregate_metrics") or {}).items()):
            lines.append(f"| {key} | {_format_value(value)} |")
        lines.append("")
        failures = [case for case in suite.get("cases", []) if not case.get("passed")]
        lines.append("### 失败案例")
        if not failures:
            lines.append("")
            lines.append("无失败案例。")
        else:
            lines.append("")
            lines.append("| Case | Query | Errors | Trace |")
            lines.append("| --- | --- | --- | --- |")
            for case in failures[:10]:
                errors = "; ".join(str(item) for item in case.get("errors", []))
                lines.append(
                    f"| {case.get('case_id')} | {case.get('query')} | {errors} | {case.get('trace_path') or ''} |"
                )
        lines.append("")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
