"""轻量本地 UI dashboard。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from kairorag.cloud.doctor import CloudDoctor, report_to_dict
from kairorag.config import KairoCloudSettings


SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
TEXT_LIMIT = 240


@dataclass(frozen=True)
class DashboardResponse:
    status: int
    body: str
    content_type: str = "text/html; charset=utf-8"


class DashboardApp:
    """可测试的 dashboard 路由对象。"""

    def __init__(self, root: str | Path | None = None, settings: KairoCloudSettings | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.settings = settings or KairoCloudSettings()

    def handle(self, raw_path: str) -> DashboardResponse:
        parsed = urlparse(raw_path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path == "/healthz":
            return DashboardResponse(200, "ok", "text/plain; charset=utf-8")
        if path == "/static/style.css":
            return DashboardResponse(200, _style(), "text/css; charset=utf-8")
        if path == "/":
            return DashboardResponse(200, self._index())
        if path == "/eval":
            return DashboardResponse(200, self._eval())
        if path == "/trace":
            limit = int(query.get("limit", ["20"])[0])
            return DashboardResponse(200, self._jsonl_page("Trace", self.settings.trace_log_path, limit=limit))
        if path == "/audit":
            limit = int(query.get("limit", ["20"])[0])
            return DashboardResponse(200, self._jsonl_page("Audit", self.settings.freshness_audit_log_path, limit=limit))
        if path == "/doctor":
            return DashboardResponse(200, self._doctor())
        return DashboardResponse(404, _layout("未找到", "<p>页面不存在。</p>"))

    def _index(self) -> str:
        manifest = self.root / self.settings.cloud_index_manifest_path
        bm25 = self.root / self.settings.cloud_bm25_index_path
        eval_dir = self.root / "results" / "eval_dashboard"
        rows = [
            ("manifest", str(manifest), manifest.exists()),
            ("BM25 index", str(bm25), bm25.exists()),
            ("trace log", str(self.root / self.settings.trace_log_path), (self.root / self.settings.trace_log_path).exists()),
            (
                "audit log",
                str(self.root / self.settings.freshness_audit_log_path),
                (self.root / self.settings.freshness_audit_log_path).exists(),
            ),
            ("eval output", str(eval_dir), eval_dir.exists()),
        ]
        table = "".join(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(path)}</td>"
            f"<td>{'存在' if exists else '缺失'}</td>"
            "</tr>"
            for name, path, exists in rows
        )
        body = f"""
<section class="notice">本 UI 仅建议本地使用，不要直接暴露到公网。</section>
<section>
  <h2>项目状态</h2>
  <table><thead><tr><th>项目</th><th>路径</th><th>状态</th></tr></thead><tbody>{table}</tbody></table>
</section>
"""
        return _layout("KairoRAG Dashboard", body)

    def _eval(self) -> str:
        payload = self._load_eval_payload()
        if not payload:
            return _layout("Eval", "<p>未找到 eval_results.json 或 eval_report.json。</p>")
        suites = payload.get("suites", [])
        rows = []
        for suite in suites:
            for key, value in (suite.get("aggregate_metrics") or {}).items():
                rows.append(
                    "<tr>"
                    f"<td>{escape(str(suite.get('suite')))}</td>"
                    f"<td>{escape(str(key))}</td>"
                    f"<td>{escape(str(value))}</td>"
                    "</tr>"
                )
        body = "<h2>Eval Metrics</h2><table><thead><tr><th>Suite</th><th>Metric</th><th>Value</th></tr></thead>"
        body += f"<tbody>{''.join(rows)}</tbody></table>"
        return _layout("Eval", body)

    def _jsonl_page(self, title: str, relative_path: str, *, limit: int) -> str:
        path = self.root / relative_path
        events = _read_jsonl_tail(path, limit=max(1, min(limit, 100)))
        rows = []
        for event in events:
            safe = _sanitize(event)
            rows.append(
                "<tr>"
                f"<td><pre>{escape(json.dumps(safe, ensure_ascii=False, indent=2))}</pre></td>"
                "</tr>"
            )
        if not rows:
            rows.append("<tr><td>暂无记录。</td></tr>")
        body = f"<p>路径：{escape(str(path))}</p><table><tbody>{''.join(rows)}</tbody></table>"
        return _layout(title, body)

    def _doctor(self) -> str:
        report = CloudDoctor(self.settings).run(live=False)
        payload = _sanitize(report_to_dict(report))
        body = f"<pre>{escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>"
        return _layout("Doctor", body)

    def _load_eval_payload(self) -> dict[str, Any] | None:
        candidates = [
            self.root / "results" / "eval_dashboard" / "eval_results.json",
            self.root / "results" / "eval_report.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        return None


def create_dashboard_app(root: str | Path | None = None, settings: KairoCloudSettings | None = None) -> DashboardApp:
    """创建可在测试中直接调用的 dashboard app。"""

    return DashboardApp(root=root, settings=settings)


def serve_dashboard(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动本地 HTTP dashboard。"""

    app = create_dashboard_app()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            response = app.handle(self.path)
            body = response.body.encode("utf-8")
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"KairoRAG UI dashboard 正在运行：http://{host}:{port}")
    server.serve_forever()


def _layout(title: str, body: str) -> str:
    nav = """
<nav>
  <a href="/">首页</a>
  <a href="/eval">Eval</a>
  <a href="/trace">Trace</a>
  <a href="/audit">Audit</a>
  <a href="/doctor">Doctor</a>
</nav>
"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  {nav}
  <main>
    <h1>{escape(title)}</h1>
    {body}
  </main>
</body>
</html>"""


def _style() -> str:
    css_path = Path(__file__).parent / "static" / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return "body{font-family:Arial,sans-serif;}"


def _read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("key", "token", "secret", "password")):
                safe[key] = "[已隐藏]"
            else:
                safe[key] = _sanitize(item)
        return safe
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        clean = SECRET_RE.sub("[已隐藏]", value)
        if len(clean) > TEXT_LIMIT:
            return clean[:TEXT_LIMIT] + "..."
        return clean
    return value
