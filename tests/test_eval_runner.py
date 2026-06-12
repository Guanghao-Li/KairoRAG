from kairorag.eval.runner import EvalRunner, run_eval_dashboard


def test_eval_runner_runs_fake_suite_and_records_aggregate_metrics():
    result = EvalRunner().run_suite("retrieval")

    assert result.suite == "retrieval"
    assert result.case_count == 2
    assert "retrieval_hit_rate" in result.aggregate_metrics
    assert result.passed_count == 2


def test_eval_runner_records_failed_case():
    result = EvalRunner().run_suite("agent")

    assert result.failed_count == 1
    assert any(case.errors for case in result.cases if not case.passed)


def test_run_eval_dashboard_writes_all_outputs(tmp_path):
    output = run_eval_dashboard(
        suite="all",
        output_dir=tmp_path,
        output_format="all",
        fake_providers=True,
        live=False,
    )

    assert len(output["written_files"]) == 3
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_dashboard.html").exists()
