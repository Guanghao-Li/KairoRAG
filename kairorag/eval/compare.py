"""Eval 结果比较和 trend JSON 写入。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


REGRESSION_RULES = {
    "retrieval_hit_rate": "decrease_bad",
    "citation_validity": "decrease_bad",
    "false_active_rate": "increase_bad",
    "unauthorized_apply_block_rate": "decrease_bad",
}


@dataclass(frozen=True)
class EvalComparison:
    baseline_path: str
    current_path: str
    metric_deltas: dict[str, float]
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    failed_cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_eval_results(baseline: str | Path | dict[str, Any], current: str | Path | dict[str, Any]) -> EvalComparison:
    """比较两份 eval JSON，并标记关键指标回归。"""

    baseline_payload, baseline_path = _load_payload(baseline)
    current_payload, current_path = _load_payload(current)
    baseline_metrics = _flatten_metrics(baseline_payload)
    current_metrics = _flatten_metrics(current_payload)
    metric_deltas = {
        key: float(current_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0))
        for key in sorted(set(baseline_metrics) | set(current_metrics))
    }
    regressions = _classify_metric_changes(metric_deltas, regression=True)
    improvements = _classify_metric_changes(metric_deltas, regression=False)
    return EvalComparison(
        baseline_path=baseline_path,
        current_path=current_path,
        metric_deltas=metric_deltas,
        regressions=regressions,
        improvements=improvements,
        failed_cases=_failed_cases(current_payload),
    )


def write_comparison_markdown(comparison: EvalComparison, path: str | Path) -> Path:
    """写出 Markdown 版 comparison 报告。"""

    lines = [
        "# KairoRAG Eval Comparison",
        "",
        f"- baseline：`{comparison.baseline_path}`",
        f"- current：`{comparison.current_path}`",
        f"- 回归数量：{len(comparison.regressions)}",
        f"- 改善数量：{len(comparison.improvements)}",
        "",
        "## 指标差异",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
    ]
    for key, delta in sorted(comparison.metric_deltas.items()):
        lines.append(f"| {key} | {delta:.4f} |")
    lines.extend(["", "## 回归摘要", ""])
    if comparison.regressions:
        for item in comparison.regressions:
            lines.append(f"- {item['metric']}：delta={item['delta']:.4f}，规则={item['rule']}")
    else:
        lines.append("未发现关键指标回归。")
    lines.extend(["", "## 失败案例 Drilldown", ""])
    if comparison.failed_cases:
        lines.extend(["| Suite | Case | Query | Errors | Trace |", "| --- | --- | --- | --- | --- |"])
        for case in comparison.failed_cases:
            trace = case.get("trace_path") or ""
            lines.append(
                f"| {case.get('suite')} | {case.get('case_id')} | {case.get('query')} | "
                f"{'; '.join(case.get('errors', []))} | {trace} |"
            )
    else:
        lines.append("当前结果没有失败案例。")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_comparison_html(comparison: EvalComparison, path: str | Path) -> Path:
    """写出无外部依赖的 HTML comparison dashboard。"""

    rows = "\n".join(
        f"<tr><td>{escape(key)}</td><td>{delta:.4f}</td></tr>"
        for key, delta in sorted(comparison.metric_deltas.items())
    )
    regression_items = "\n".join(
        f"<li>{escape(str(item['metric']))}: {float(item['delta']):.4f}</li>" for item in comparison.regressions
    ) or "<li>未发现关键指标回归。</li>"
    failed_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(case.get('suite')))}</td>"
        f"<td>{escape(str(case.get('case_id')))}</td>"
        f"<td>{escape(str(case.get('query')))}</td>"
        f"<td>{escape('; '.join(case.get('errors', [])))}</td>"
        f"<td>{escape(str(case.get('trace_path') or ''))}</td>"
        "</tr>"
        for case in comparison.failed_cases
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>KairoRAG Eval Comparison</title>
  <style>
    body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f7f8;color:#20262d;}}
    main{{max-width:1120px;margin:0 auto;padding:28px;}}
    table{{width:100%;border-collapse:collapse;background:#fff;margin:12px 0 24px;}}
    th,td{{border:1px solid #d8dee4;padding:8px;text-align:left;font-size:14px;vertical-align:top;}}
    th{{background:#eef2f5;}}
    .danger{{color:#a63232;font-weight:700;}}
  </style>
</head>
<body><main>
  <h1>KairoRAG Eval Comparison</h1>
  <p>Baseline: {escape(comparison.baseline_path)}<br>Current: {escape(comparison.current_path)}</p>
  <p class="danger">回归数量：{len(comparison.regressions)}</p>
  <h2>指标差异</h2>
  <table><thead><tr><th>Metric</th><th>Delta</th></tr></thead><tbody>{rows}</tbody></table>
  <h2>回归摘要</h2>
  <ul>{regression_items}</ul>
  <h2>失败案例 Drilldown</h2>
  <table><thead><tr><th>Suite</th><th>Case</th><th>Query</th><th>Errors</th><th>Trace</th></tr></thead>
  <tbody>{failed_rows}</tbody></table>
</main></body></html>"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_trend_json(comparison: EvalComparison, path: str | Path, *, tag: str | None = None) -> Path:
    """向 trend JSON 追加一次 comparison 摘要。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
    else:
        payload = []
    if isinstance(payload, dict):
        entries = list(payload.get("entries", []))
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    entries.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": tag,
            "baseline_path": comparison.baseline_path,
            "current_path": comparison.current_path,
            "metric_deltas": comparison.metric_deltas,
            "regression_count": len(comparison.regressions),
            "improvement_count": len(comparison.improvements),
        }
    )
    output_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _load_payload(value: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        return value, "<memory>"
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8-sig")), str(path)


def _flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    suites = payload.get("suites")
    if isinstance(suites, list):
        for suite in suites:
            for key, value in (suite.get("aggregate_metrics") or {}).items():
                if isinstance(value, int | float):
                    metrics[str(key)] = float(value)
    for key, value in (payload.get("aggregate_metrics") or {}).items():
        if isinstance(value, int | float):
            metrics[str(key)] = float(value)
    return metrics


def _classify_metric_changes(metric_deltas: dict[str, float], *, regression: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for metric, rule in REGRESSION_RULES.items():
        delta = metric_deltas.get(metric)
        if delta is None or delta == 0:
            continue
        is_regression = (rule == "decrease_bad" and delta < 0) or (rule == "increase_bad" and delta > 0)
        if is_regression is regression:
            items.append({"metric": metric, "delta": delta, "rule": rule})
    return items


def _failed_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for suite in payload.get("suites", []):
        suite_name = suite.get("suite")
        for case in suite.get("cases", []):
            if not case.get("passed"):
                cases.append({"suite": suite_name, **case})
    return cases
