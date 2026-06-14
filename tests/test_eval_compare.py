import json

from kairorag.eval.compare import (
    compare_eval_results,
    write_comparison_html,
    write_comparison_markdown,
    write_trend_json,
)


def _payload(retrieval_hit_rate, false_active_rate):
    return {
        "suites": [
            {
                "suite": "retrieval",
                "aggregate_metrics": {"retrieval_hit_rate": retrieval_hit_rate},
                "cases": [],
            },
            {
                "suite": "freshness",
                "aggregate_metrics": {"false_active_rate": false_active_rate},
                "cases": [
                    {
                        "case_id": "fresh-1",
                        "query": "岗位还开放吗",
                        "passed": False,
                        "errors": ["状态错误"],
                        "trace_path": "trace/fresh-1.json",
                    }
                ],
            },
        ]
    }


def test_eval_compare_detects_regression_and_writes_outputs(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_payload(1.0, 0.0)), encoding="utf-8")
    current.write_text(json.dumps(_payload(0.5, 0.2)), encoding="utf-8")

    comparison = compare_eval_results(baseline, current)
    md = write_comparison_markdown(comparison, tmp_path / "compare.md")
    html = write_comparison_html(comparison, tmp_path / "compare.html")
    trend = write_trend_json(comparison, tmp_path / "trend.json", tag="current")

    assert comparison.metric_deltas["retrieval_hit_rate"] == -0.5
    assert {item["metric"] for item in comparison.regressions} == {"retrieval_hit_rate", "false_active_rate"}
    assert "fresh-1" in md.read_text(encoding="utf-8")
    assert "retrieval_hit_rate" in html.read_text(encoding="utf-8")
    assert json.loads(trend.read_text(encoding="utf-8"))["entries"][0]["tag"] == "current"
