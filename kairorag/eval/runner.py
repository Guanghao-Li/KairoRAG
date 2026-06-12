"""离线 eval runner。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from kairorag.eval.datasets import EvalCase, EvalCaseResult, EvalSuiteResult, load_fixture_cases
from kairorag.eval.dashboard import write_html_dashboard
from kairorag.eval.fake_providers import FakeEvalProviders
from kairorag.eval.metrics import (
    active_precision,
    allowed_field_violation_rate,
    audit_log_created_rate,
    chunk_read_before_answer_rate,
    citation_validity,
    closed_recall,
    dry_run_default_rate,
    empty_rerank_rate,
    evidence_recall,
    false_active_rate,
    final_answer_from_grounded_generator_rate,
    invalid_citation_rate,
    invalid_tool_call_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    no_evidence_refusal_rate,
    planning_success_rate,
    rerank_improvement_rate,
    rerank_mrr_delta,
    retrieval_hit_rate,
    snippet_leakage_rate,
    stale_detection_rate,
    tool_budget_exceeded_rate,
    unauthorized_apply_block_rate,
    verification_accuracy,
)
from kairorag.eval.report import write_json_report, write_markdown_report


ALL_SUITES = ["retrieval", "groundedness", "agent", "freshness", "reranker", "writeback"]


class EvalRunner:
    """使用 fake providers 运行 cloud-native 离线 eval suite。"""

    def __init__(self, providers: FakeEvalProviders | None = None) -> None:
        self.providers = providers or FakeEvalProviders()

    def run_suite(self, suite: str, *, live: bool = False) -> EvalSuiteResult:
        """运行一个 suite；默认不允许 live 外部调用。"""

        if live:
            raise ValueError("live eval 默认未实现；请使用 --fake-providers 运行离线评测。")
        if suite == "retrieval":
            return self._run_retrieval(load_fixture_cases(suite))
        if suite == "groundedness":
            return self._run_groundedness(load_fixture_cases(suite))
        if suite == "agent":
            return self._run_agent(load_fixture_cases(suite))
        if suite == "freshness":
            return self._run_freshness(load_fixture_cases(suite))
        if suite == "reranker":
            return self._run_reranker(load_fixture_cases(suite))
        if suite == "writeback":
            return self._run_writeback(load_fixture_cases(suite))
        raise ValueError(f"不支持的 eval suite：{suite}")

    def run(self, suite: str = "all", *, live: bool = False) -> dict[str, Any]:
        """运行一个或全部 suite，并返回 dashboard payload。"""

        suites = ALL_SUITES if suite == "all" else [suite]
        results = [self.run_suite(item, live=live) for item in suites]
        return {
            "suites": [asdict(item) for item in results],
            "aggregate": {
                "suite_count": len(results),
                "case_count": sum(item.case_count for item in results),
                "passed_count": sum(item.passed_count for item in results),
                "failed_count": sum(item.failed_count for item in results),
            },
        }

    def _run_retrieval(self, cases: list[EvalCase]) -> EvalSuiteResult:
        raw = [self.providers.retrieval_result(case) for case in cases]
        metrics = {
            "retrieval_hit_rate": retrieval_hit_rate(raw),
            "evidence_recall": evidence_recall(raw),
            "mrr": mean_reciprocal_rank(raw),
            "nDCG@5": ndcg_at_k(raw, k=5),
            "read_chunk_recall": evidence_recall(raw),
            "context_token_usage": sum(float(item.get("context_tokens", 0)) for item in raw) / len(raw),
        }
        case_results = []
        for case, item in zip(cases, raw):
            passed = bool(set(case.expected_chunk_ids) & set(item["retrieved_chunk_ids"]))
            case_results.append(_case_result(case, passed, {"hit": passed}, [] if passed else ["未命中期望 chunk。"]))
        return _suite_result("retrieval", metrics, case_results)

    def _run_groundedness(self, cases: list[EvalCase]) -> EvalSuiteResult:
        answers = [self.providers.grounded_answer(case) for case in cases]
        metrics = {
            "citation_validity": sum(citation_validity(item) for item in answers) / len(answers),
            "invalid_citation_rate": invalid_citation_rate(answers),
            "no_evidence_refusal_rate": no_evidence_refusal_rate(answers),
            "snippet_leakage_rate": snippet_leakage_rate(answers),
        }
        case_results = []
        for case, answer in zip(cases, answers):
            passed = citation_validity(answer) == 1.0 and not answer.get("used_search_snippet_as_evidence")
            errors = [] if passed else ["citation 或 snippet guardrail 失败。"]
            case_results.append(_case_result(case, passed, {"citation_validity": citation_validity(answer)}, errors))
        return _suite_result("groundedness", metrics, case_results)

    def _run_agent(self, cases: list[EvalCase]) -> EvalSuiteResult:
        raw = [self.providers.agent_result(case) for case in cases]
        metrics = {
            "planning_success_rate": planning_success_rate(raw),
            "invalid_tool_call_rate": invalid_tool_call_rate(raw),
            "tool_budget_exceeded_rate": tool_budget_exceeded_rate(raw),
            "chunk_read_before_answer_rate": chunk_read_before_answer_rate(raw),
            "final_answer_from_grounded_generator_rate": final_answer_from_grounded_generator_rate(raw),
        }
        case_results = []
        for case, item in zip(cases, raw):
            passed = bool(item["planning_success"]) and int(item["invalid_tool_calls"]) == 0
            errors = [] if passed else ["Agent planning 或工具调用不满足 guardrail。"]
            case_results.append(_case_result(case, passed, item, errors))
        return _suite_result("agent", metrics, case_results)

    def _run_freshness(self, cases: list[EvalCase]) -> EvalSuiteResult:
        raw = [self.providers.freshness_result(case) for case in cases]
        metrics = {
            "verification_accuracy": verification_accuracy(raw),
            "active_precision": active_precision(raw),
            "closed_recall": closed_recall(raw),
            "stale_detection_rate": stale_detection_rate(raw),
            "false_active_rate": false_active_rate(raw),
        }
        case_results = []
        for case, item in zip(cases, raw):
            passed = item["predicted_status"] == item["expected_status"]
            case_results.append(_case_result(case, passed, item, [] if passed else ["freshness status 预测错误。"]))
        return _suite_result("freshness", metrics, case_results)

    def _run_reranker(self, cases: list[EvalCase]) -> EvalSuiteResult:
        before: list[dict[str, Any]] = []
        after: list[dict[str, Any]] = []
        for case in cases:
            before_item, after_item = self.providers.rerank_pair(case)
            before.append(before_item)
            after.append(after_item)
        metrics = {
            "rerank_improvement_rate": rerank_improvement_rate(before, after),
            "rerank_mrr_delta": rerank_mrr_delta(before, after),
            "rerank_latency_ms": sum(float(item.get("latency_ms", 0)) for item in after) / len(after),
            "empty_rerank_rate": empty_rerank_rate(after),
        }
        case_results = []
        for case, before_item, after_item in zip(cases, before, after):
            passed = mean_reciprocal_rank([after_item]) >= mean_reciprocal_rank([before_item])
            case_results.append(_case_result(case, passed, {"before": before_item, "after": after_item}, [] if passed else ["rerank 后排序变差。"]))
        return _suite_result("reranker", metrics, case_results)

    def _run_writeback(self, cases: list[EvalCase]) -> EvalSuiteResult:
        raw = [self.providers.writeback_result(case) for case in cases]
        metrics = {
            "unauthorized_apply_block_rate": unauthorized_apply_block_rate(raw),
            "dry_run_default_rate": dry_run_default_rate(raw),
            "allowed_field_violation_rate": allowed_field_violation_rate(raw),
            "audit_log_created_rate": audit_log_created_rate(raw),
        }
        case_results = []
        for case, item in zip(cases, raw):
            passed = (not item["requested_apply"] or item["blocked"]) and not item["allowed_field_violation"]
            errors = [] if passed else ["写回安全策略未阻断未授权请求。"]
            case_results.append(_case_result(case, passed, item, errors))
        return _suite_result("writeback", metrics, case_results)


def run_eval_dashboard(
    *,
    suite: str,
    output_dir: Path,
    output_format: str = "all",
    fake_providers: bool = True,
    live: bool = False,
) -> dict[str, Any]:
    """运行 eval 并按需写出 JSON、Markdown、HTML dashboard。"""

    if live and not fake_providers:
        raise ValueError("live eval 尚未接入真实 provider；当前阶段请使用 fake providers。")
    runner = EvalRunner()
    payload = runner.run(suite=suite, live=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if output_format in {"json", "all"}:
        written.append(str(write_json_report(payload, output_dir / "eval_results.json")))
    if output_format in {"markdown", "all"}:
        written.append(str(write_markdown_report(payload, output_dir / "eval_report.md")))
    if output_format in {"html", "all"}:
        written.append(str(write_html_dashboard(payload, output_dir / "eval_dashboard.html")))
    return {"result": payload, "written_files": written}


def _case_result(case: EvalCase, passed: bool, metrics: dict[str, Any], errors: list[str]) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=passed,
        metrics=metrics,
        errors=errors,
        trace_path=None,
    )


def _suite_result(suite: str, metrics: dict[str, Any], cases: list[EvalCaseResult]) -> EvalSuiteResult:
    passed_count = sum(1 for item in cases if item.passed)
    return EvalSuiteResult(
        suite=suite,
        case_count=len(cases),
        passed_count=passed_count,
        failed_count=len(cases) - passed_count,
        aggregate_metrics=metrics,
        cases=cases,
    )
