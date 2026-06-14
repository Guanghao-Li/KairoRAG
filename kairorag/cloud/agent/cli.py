"""Cloud LLM Agent CLI。"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from kairorag.cloud.observability import safe_json_dumps
from kairorag.cloud.runtime import build_cloud_agent_runtime
from kairorag.config import KairoCloudSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 KairoRAG cloud LLM autonomous agent。")
    parser.add_argument("query", help="要询问的问题。")
    parser.add_argument("--include-trace", action="store_true", help="打印简洁 trace。")
    parser.add_argument("--max-tool-calls", type=int, default=None, help="覆盖最大工具调用次数。")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖检索候选数量上限。")
    parser.add_argument("--source-type", default=None, help="按 source_type 过滤。")
    parser.add_argument("--company", default=None, help="按 company 过滤。")
    parser.add_argument("--verify-freshness", action="store_true", help="要求启用岗位 freshness verification。")
    parser.add_argument("--show-verification", action="store_true", help="打印岗位 verification evidence 摘要。")
    parser.add_argument("--web-provider", choices=["tavily", "serpapi", "bing"], default=None, help="覆盖 Web Search。")
    parser.add_argument(
        "--reranker",
        choices=["base_score", "cohere", "jina", "voyage", "openai_listwise", "cross_encoder"],
        default=None,
        help="覆盖 RERANKER_PROVIDER。",
    )
    parser.add_argument("--rerank-top-k", type=int, default=None, help="覆盖 RERANK_TOP_K。")
    parser.add_argument("--apply-freshness-update", action="store_true", help="请求 freshness metadata 写回。")
    parser.add_argument("--dry-run", choices=["true", "false"], default=None, help="是否 dry-run，默认 true。")
    parser.add_argument("--yes", action="store_true", help="确认执行真实 destructive apply。")
    parser.add_argument(
        "--approval-policy",
        choices=["deny", "dry_run", "require_confirmation", "allow"],
        default=None,
        help="覆盖审批策略。",
    )
    parser.add_argument("--audit-log", default=None, help="覆盖 freshness audit JSONL 路径。")
    parser.add_argument("--trace-output", default=None, help="把本次 trace 保存为 JSON 文件。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = KairoCloudSettings()
    if args.web_provider:
        settings.web_search_provider = args.web_provider
    if args.reranker:
        settings.reranker_provider = args.reranker
    if args.rerank_top_k is not None:
        settings.rerank_top_k = args.rerank_top_k
    if args.audit_log:
        settings.freshness_audit_log_path = args.audit_log
    if args.approval_policy:
        settings.approval_policy = args.approval_policy
    if args.dry_run is not None:
        settings.qdrant_metadata_write_dry_run = _parse_bool(args.dry_run)
    if args.verify_freshness and not settings.job_freshness_enabled:
        print("错误：JOB_FRESHNESS_ENABLED=false，无法执行 --verify-freshness。")
        return 2

    freshness_apply_authorized = bool(args.apply_freshness_update and args.yes)
    runtime = _build_agent_runtime(settings, freshness_apply_authorized=freshness_apply_authorized)
    if args.max_tool_calls is not None:
        runtime.settings.max_tool_calls = args.max_tool_calls
    if args.top_k is not None:
        runtime.settings.max_search_results = args.top_k
        runtime.settings.max_chunks_to_read = min(runtime.settings.max_chunks_to_read, args.top_k)
    result = runtime.llm_agent.ask(
        args.query,
        filters=_build_filters(source_type=args.source_type, company=args.company),
    )
    _print_result(result, include_trace=args.include_trace, show_verification=args.show_verification)
    print()
    print(f"Reranker：{runtime.settings.reranker_provider}")
    _print_rerank_metrics(result.metrics)
    _print_freshness_update_summary(getattr(result, "trace", []))
    if args.trace_output:
        _write_trace_output(args.trace_output, result)
        print(f"Trace output：{args.trace_output}")
    return 0


def _build_filters(*, source_type: str | None, company: str | None) -> dict[str, str] | None:
    filters: dict[str, str] = {}
    if source_type:
        filters["source_type"] = source_type
    if company:
        filters["company"] = company
    return filters or None


def _build_agent_runtime(settings: KairoCloudSettings, *, freshness_apply_authorized: bool) -> Any:
    signature = inspect.signature(build_cloud_agent_runtime)
    if "freshness_apply_authorized" in signature.parameters:
        return build_cloud_agent_runtime(settings, freshness_apply_authorized=freshness_apply_authorized)
    return build_cloud_agent_runtime(settings)


def _print_result(result: Any, *, include_trace: bool, show_verification: bool = False) -> None:
    print("最终回答：")
    print(result.answer)
    print()
    print("引用：")
    if result.citations:
        for citation in result.citations:
            print(f"- {citation.chunk_id} | {citation.title} | {citation.evidence}")
    else:
        print("- 无")
    print()
    print("Metrics：")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=_json_default))
    print()
    if show_verification:
        _print_verification(getattr(result, "verification_results", []))
        print()
    if include_trace:
        print("Trace：")
        for step in result.trace:
            detail = _shorten_detail(step.detail)
            print(f"- {step.step}: {json.dumps(detail, ensure_ascii=False, default=_json_default)}")
    else:
        print(f"Trace：{len(result.trace)} 个步骤，使用 --include-trace 查看详情。")


def _print_verification(results: list[Any]) -> None:
    print("Verification：")
    if not results:
        print("- 无")
        return
    for result in results:
        print(f"- status={result.status} confidence={result.confidence} action={result.recommended_action}")
        print(f"  reason={result.reason}")
        for evidence in result.evidence[:3]:
            print(
                "  evidence="
                f"{evidence.source} | {evidence.signal} | {evidence.url} | {_preview(evidence.snippet, 120)}"
            )


def _shorten_detail(detail: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(detail, ensure_ascii=False, default=_json_default)
    if len(text) <= 600:
        return detail
    return {"summary": text[:600] + "..."}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "是"}


def _print_rerank_metrics(metrics: dict[str, Any]) -> None:
    relevant = {
        key: value
        for key, value in metrics.items()
        if "rerank" in key or key in {"reranker_provider", "retrieval_reranker_provider"}
    }
    if relevant:
        print("Rerank metrics：")
        print(json.dumps(relevant, ensure_ascii=False, indent=2, default=_json_default))


def _print_freshness_update_summary(trace: list[Any]) -> None:
    updates = [step for step in trace if getattr(step, "step", "") == "freshness_update"]
    if not updates:
        return
    print("Freshness update：")
    for step in updates:
        detail = getattr(step, "detail", {})
        print(json.dumps(_shorten_detail(detail), ensure_ascii=False, default=_json_default))


def _write_trace_output(path: str, result: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": getattr(result, "query", None),
        "trace": [asdict(step) if is_dataclass(step) else step for step in getattr(result, "trace", [])],
        "metrics": getattr(result, "metrics", {}),
    }
    output_path.write_text(safe_json_dumps(payload), encoding="utf-8")


def _preview(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
