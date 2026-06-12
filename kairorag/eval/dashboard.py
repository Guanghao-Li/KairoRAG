"""Eval HTML dashboard 生成。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


def write_html_dashboard(result: Any, path: str | Path) -> Path:
    """写出不依赖外部 CDN 的静态 HTML dashboard。"""

    payload = _to_jsonable(result)
    suites = payload.get("suites", [payload] if payload.get("suite") else [])
    case_count = sum(int(item.get("case_count", 0)) for item in suites)
    passed_count = sum(int(item.get("passed_count", 0)) for item in suites)
    failed_count = sum(int(item.get("failed_count", 0)) for item in suites)
    html = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>KairoRAG Eval Dashboard</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f7f7f4;color:#1d252c;}",
        "main{max-width:1180px;margin:0 auto;padding:28px;}",
        "h1{font-size:28px;margin:0 0 18px;}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0 24px;}",
        ".card{background:#fff;border:1px solid #d9ded8;border-radius:6px;padding:14px;}",
        ".card strong{display:block;font-size:24px;margin-top:6px;}",
        "table{width:100%;border-collapse:collapse;background:#fff;margin:12px 0 24px;}",
        "th,td{border:1px solid #d9ded8;padding:8px;text-align:left;font-size:14px;vertical-align:top;}",
        "th{background:#e9eee8;}",
        ".failed{color:#a63232;font-weight:700;}",
        ".passed{color:#1f7a4d;font-weight:700;}",
        "section{margin:24px 0;}",
        "</style>",
        "</head>",
        "<body><main>",
        "<h1>KairoRAG Eval Dashboard</h1>",
        f"<p>生成时间：{escape(datetime.now(timezone.utc).isoformat())}</p>",
        '<div class="cards">',
        f'<div class="card">Suite<strong>{len(suites)}</strong></div>',
        f'<div class="card">Case<strong>{case_count}</strong></div>',
        f'<div class="card">Passed<strong>{passed_count}</strong></div>',
        f'<div class="card">Failed<strong>{failed_count}</strong></div>',
        "</div>",
    ]
    html.append("<section><h2>Suite 指标</h2>")
    html.append("<table><thead><tr><th>Suite</th><th>Metric</th><th>Value</th></tr></thead><tbody>")
    for suite in suites:
        for key, value in sorted((suite.get("aggregate_metrics") or {}).items()):
            html.append(
                "<tr>"
                f"<td>{escape(str(suite.get('suite')))}</td>"
                f"<td>{escape(str(key))}</td>"
                f"<td>{escape(_format_value(value))}</td>"
                "</tr>"
            )
    html.append("</tbody></table></section>")
    html.append("<section><h2>Case 明细</h2>")
    html.append("<table><thead><tr><th>Suite</th><th>Case</th><th>Query</th><th>Status</th><th>Errors</th></tr></thead><tbody>")
    for suite in suites:
        for case in suite.get("cases", []):
            status_class = "passed" if case.get("passed") else "failed"
            status = "通过" if case.get("passed") else "失败"
            errors = "; ".join(str(item) for item in case.get("errors", []))
            html.append(
                "<tr>"
                f"<td>{escape(str(suite.get('suite')))}</td>"
                f"<td>{escape(str(case.get('case_id')))}</td>"
                f"<td>{escape(str(case.get('query')))}</td>"
                f'<td class="{status_class}">{status}</td>'
                f"<td>{escape(errors)}</td>"
                "</tr>"
            )
    html.append("</tbody></table></section>")
    html.append("<section><h2>Failures</h2>")
    failures = [
        (suite.get("suite"), case)
        for suite in suites
        for case in suite.get("cases", [])
        if not case.get("passed")
    ]
    if not failures:
        html.append("<p>无失败案例。</p>")
    else:
        html.append("<ul>")
        for suite_name, case in failures[:20]:
            errors = "; ".join(str(item) for item in case.get("errors", []))
            html.append(
                "<li>"
                f"{escape(str(suite_name))} / {escape(str(case.get('case_id')))}：{escape(errors)}"
                "</li>"
            )
        html.append("</ul>")
    html.append("</section></main></body></html>")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html), encoding="utf-8")
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
