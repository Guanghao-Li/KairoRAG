import json

from kairorag.eval.dashboard import write_html_dashboard
from kairorag.eval.runner import EvalRunner
from kairorag.eval.report import write_json_report, write_markdown_report


def test_eval_report_generates_json_markdown_and_html(tmp_path):
    payload = EvalRunner().run("agent")

    json_path = write_json_report(payload, tmp_path / "eval_results.json")
    md_path = write_markdown_report(payload, tmp_path / "eval_report.md")
    html_path = write_html_dashboard(payload, tmp_path / "eval_dashboard.html")

    assert json.loads(json_path.read_text(encoding="utf-8"))["aggregate"]["suite_count"] == 1
    markdown = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "agent-2" in markdown
    assert "planning_success_rate" in html
    assert "agent" in html
    assert "http://" not in html
    assert "https://" not in html
