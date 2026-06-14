"""KairoRAG 统一 cloud-native CLI。"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from kairorag.cloud import index as index_cli
from kairorag.cloud import query_cli
from kairorag.cloud.agent import cli as agent_cli
from kairorag.cloud.doctor import CloudDoctor, format_doctor_report, report_to_json, settings_summary
from kairorag.cloud.freshness import CloudFreshnessUpdater, CloudJobFreshnessVerifier
from kairorag.config import KairoCloudSettings
from kairorag.providers import build_web_search_provider


RERANKER_CHOICES = ["base_score", "cohere", "jina", "voyage", "openai_listwise", "cross_encoder"]
WEB_PROVIDER_CHOICES = ["tavily", "serpapi", "bing"]
CHECK_CHOICES = ["qdrant", "openai", "websearch", "reranker", "all"]
SUITE_CHOICES = ["retrieval", "groundedness", "agent", "freshness", "reranker", "writeback", "all"]
FORMAT_CHOICES = ["json", "markdown", "html", "all"]
APPROVAL_POLICY_CHOICES = ["deny", "dry_run", "require_confirmation", "allow"]


def build_parser() -> argparse.ArgumentParser:
    """构建统一 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="kairo",
        description="KairoRAG cloud-native 统一 CLI。",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="命令")

    index_parser = subparsers.add_parser("index", help="构建 cloud index。")
    index_parser.add_argument("--recreate", action="store_true", help="重建 Qdrant collection。")
    index_parser.add_argument("--batch-size", type=int, default=None, help="覆盖 CLOUD_CHUNK_BATCH_SIZE。")
    index_parser.add_argument("--manifest-path", default=None, help="覆盖 manifest 输出路径。")
    index_parser.add_argument("--bm25-path", default=None, help="覆盖 BM25 JSON 输出路径。")
    index_parser.set_defaults(handler=_run_index)

    query_parser = subparsers.add_parser("query", help="执行 cloud query，默认使用 LLM autonomous agent。")
    _add_query_like_arguments(query_parser, include_baseline=True)
    query_parser.set_defaults(handler=_run_query)

    agent_parser = subparsers.add_parser("agent", help="显式执行 LLM autonomous agent。")
    _add_query_like_arguments(agent_parser, include_baseline=False)
    agent_parser.add_argument("--max-tool-calls", type=int, default=None, help="覆盖最大工具调用次数。")
    agent_parser.set_defaults(handler=_run_agent)

    verify_parser = subparsers.add_parser("verify", help="单独执行岗位 freshness verification。")
    verify_parser.add_argument("--chunk-id", default=None, help="chunk id。")
    verify_parser.add_argument("--company", default=None, help="公司名称。")
    verify_parser.add_argument("--title", default=None, help="岗位标题。")
    verify_parser.add_argument("--job-id", default=None, help="岗位 id。")
    verify_parser.add_argument("--original-url", default=None, help="原始岗位 URL。")
    verify_parser.add_argument("--query", default=None, help="用于验证的搜索 query。")
    verify_parser.add_argument("--show-evidence", action="store_true", help="打印 evidence 明细。")
    verify_parser.add_argument("--apply", action="store_true", help="请求执行 metadata 写回。")
    verify_parser.add_argument("--dry-run", action="store_true", help="强制 dry-run，默认不真实写回。")
    verify_parser.add_argument("--yes", action="store_true", help="确认执行真实 destructive apply。")
    verify_parser.add_argument("--approval-policy", choices=APPROVAL_POLICY_CHOICES, default=None, help="覆盖审批策略。")
    verify_parser.set_defaults(handler=_run_verify)

    eval_parser = subparsers.add_parser("eval", help="运行离线 eval dashboard 或比较两次结果。")
    eval_parser.add_argument("--suite", choices=SUITE_CHOICES, default="all", help="选择 eval suite。")
    eval_parser.add_argument("--output-dir", default="results/eval_dashboard", help="报告输出目录。")
    eval_parser.add_argument("--format", choices=FORMAT_CHOICES, default="all", help="报告格式。")
    eval_parser.add_argument("--fake-providers", action="store_true", help="使用 fake providers，默认开启。")
    eval_parser.add_argument("--live", action="store_true", help="允许 live eval，默认不调用外部服务。")
    eval_parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"), help="比较两次 eval JSON。")
    eval_parser.add_argument("--tag", default=None, help="为本次 eval 或 comparison 附加标签。")
    eval_parser.set_defaults(handler=_run_eval)

    doctor_parser = subparsers.add_parser("doctor", help="检查环境、依赖配置或发布卫生。")
    doctor_parser.add_argument("--live", action="store_true", help="允许轻量 live healthcheck。")
    doctor_parser.add_argument("--json", action="store_true", help="输出 JSON 报告。")
    doctor_parser.add_argument("--check", choices=CHECK_CHOICES, default="all", help="选择检查项。")
    doctor_parser.add_argument("--repo", action="store_true", help="执行本地仓库卫生检查，不调用外部服务。")
    doctor_parser.add_argument("--release", action="store_true", help="执行发布前检查清单和仓库卫生检查。")
    doctor_parser.add_argument("--run-checks", action="store_true", help="在 --release 下实际运行 pytest 和 ruff。")
    doctor_parser.set_defaults(handler=_run_doctor)

    ui_parser = subparsers.add_parser("ui", help="启动本地 UI dashboard。")
    ui_parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机。")
    ui_parser.add_argument("--port", type=int, default=8000, help="监听端口。")
    ui_parser.set_defaults(handler=_run_ui)

    config_parser = subparsers.add_parser("config", help="打印脱敏配置摘要。")
    config_parser.add_argument("--json", action="store_true", help="输出 JSON。")
    config_parser.add_argument("--show-paths", action="store_true", help="显示 manifest、BM25、trace、audit 路径。")
    config_parser.set_defaults(handler=_run_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


def _add_query_like_arguments(parser: argparse.ArgumentParser, *, include_baseline: bool) -> None:
    parser.add_argument("query", help="要询问的问题。")
    if include_baseline:
        parser.add_argument("--baseline", action="store_true", help="使用固定 cloud RAG baseline。")
    parser.add_argument("--include-trace", action="store_true", help="打印 trace。")
    parser.add_argument("--trace-output", default=None, help="把本次 trace 保存为 JSON 文件。")
    parser.add_argument("--top-k", type=int, default=None, help="检索候选数量上限。")
    parser.add_argument("--source-type", default=None, help="按 source_type 过滤。")
    parser.add_argument("--company", default=None, help="按 company 过滤。")
    parser.add_argument("--reranker", choices=RERANKER_CHOICES, default=None, help="覆盖 RERANKER_PROVIDER。")
    parser.add_argument("--rerank-top-k", type=int, default=None, help="覆盖 RERANK_TOP_K。")
    parser.add_argument("--verify-freshness", action="store_true", help="要求启用岗位 freshness verification。")
    parser.add_argument("--show-verification", action="store_true", help="打印岗位 verification evidence 摘要。")
    parser.add_argument("--apply-freshness-update", action="store_true", help="请求 freshness metadata 写回。")
    parser.add_argument("--dry-run", choices=["true", "false"], default=None, help="是否 dry-run，默认 true。")
    parser.add_argument("--yes", action="store_true", help="确认执行真实 destructive apply。")
    parser.add_argument("--approval-policy", choices=APPROVAL_POLICY_CHOICES, default=None, help="覆盖审批策略。")
    parser.add_argument("--web-provider", choices=WEB_PROVIDER_CHOICES, default=None, help="覆盖 Web Search provider。")
    parser.add_argument("--audit-log", default=None, help="覆盖 freshness audit JSONL 路径。")


def _run_index(args: argparse.Namespace) -> int:
    return index_cli.main(_argv_from_args(args, ["recreate", "batch_size", "manifest_path", "bm25_path"]))


def _run_query(args: argparse.Namespace) -> int:
    argv = [args.query] + _argv_from_args(
        args,
        [
            "baseline",
            "include_trace",
            "trace_output",
            "top_k",
            "source_type",
            "company",
            "reranker",
            "rerank_top_k",
            "verify_freshness",
            "show_verification",
            "apply_freshness_update",
            "dry_run",
            "yes",
            "approval_policy",
            "web_provider",
            "audit_log",
        ],
    )
    return query_cli.main(argv)


def _run_agent(args: argparse.Namespace) -> int:
    argv = [args.query] + _argv_from_args(
        args,
        [
            "include_trace",
            "trace_output",
            "top_k",
            "source_type",
            "company",
            "reranker",
            "rerank_top_k",
            "verify_freshness",
            "show_verification",
            "apply_freshness_update",
            "dry_run",
            "yes",
            "approval_policy",
            "web_provider",
            "audit_log",
            "max_tool_calls",
        ],
    )
    return agent_cli.main(argv)


def _run_verify(args: argparse.Namespace) -> int:
    settings = KairoCloudSettings()
    if args.approval_policy:
        settings.approval_policy = args.approval_policy
    provider = build_web_search_provider(settings)
    verifier = CloudJobFreshnessVerifier(settings, provider)
    result = verifier.verify_by_query(
        args.query or "",
        company=args.company,
        title=args.title,
        job_id=args.job_id,
        original_url=args.original_url,
        chunk_id=args.chunk_id,
    )
    print("岗位 freshness verification 结果：")
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    if args.show_evidence:
        print("Evidence：")
        for evidence in result.evidence:
            print(f"- {evidence.source} | {evidence.signal} | {evidence.url} | {evidence.snippet}")
    if args.apply:
        from kairorag.cloud.runtime import build_cloud_runtime

        runtime = build_cloud_runtime(settings)
        updater = CloudFreshnessUpdater(settings)
        update = updater.apply_update(
            result,
            vector_store=runtime.vector_store,
            dry_run=args.dry_run or not args.apply,
            user_confirmed=args.yes,
            agent_initiated=False,
            approval_policy=args.approval_policy,
        )
        print("写回结果：")
        print(json.dumps(_jsonable(update), ensure_ascii=False, indent=2))
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    if args.compare:
        from kairorag.eval.compare import (
            compare_eval_results,
            write_comparison_html,
            write_comparison_markdown,
            write_trend_json,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison = compare_eval_results(args.compare[0], args.compare[1])
        md_path = write_comparison_markdown(comparison, output_dir / "eval_comparison.md")
        html_path = write_comparison_html(comparison, output_dir / "eval_comparison.html")
        trend_path = write_trend_json(comparison, output_dir / "trend.json", tag=args.tag)
        print("Eval comparison 已生成：")
        for path in (md_path, html_path, trend_path):
            print(f"- {path}")
        return 0

    from kairorag.eval.runner import run_eval_dashboard

    result = run_eval_dashboard(
        suite=args.suite,
        output_dir=Path(args.output_dir),
        output_format=args.format,
        fake_providers=True if not args.live else args.fake_providers,
        live=args.live,
        tag=args.tag,
    )
    print("Eval dashboard 已生成：")
    for path in result["written_files"]:
        print(f"- {path}")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    if args.repo or args.release:
        from kairorag.release import RepoHygieneChecker, build_release_checklist

        report = RepoHygieneChecker(Path.cwd()).run()
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            if args.release:
                print("KairoRAG 发布前检查")
                print(build_release_checklist())
            print(_format_repo_hygiene_report(report))
            if args.release and args.run_checks:
                return _run_release_commands() or (0 if report.ok else 1)
        return 0 if report.ok else 1

    doctor = CloudDoctor(KairoCloudSettings())
    report = doctor.run(live=args.live, checks=[args.check])
    if args.json:
        print(report_to_json(report))
    else:
        print(format_doctor_report(report))
    return 0 if report.ok else 1


def _run_ui(args: argparse.Namespace) -> int:
    from kairorag.ui.app import serve_dashboard

    if args.host == "0.0.0.0":
        print("安全提醒：UI dashboard 没有内置身份认证，请不要直接暴露到公网。")
    serve_dashboard(host=args.host, port=args.port)
    return 0


def _run_config(args: argparse.Namespace) -> int:
    summary = settings_summary(KairoCloudSettings(), show_paths=args.show_paths)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("KairoRAG 配置摘要：")
        for key, value in summary.items():
            print(f"- {key}: {value}")
    return 0


def _format_repo_hygiene_report(report: Any) -> str:
    lines = ["KairoRAG Repo Hygiene 报告", f"整体状态：{'通过' if report.ok else '失败'}"]
    if not report.issues:
        lines.append("- 未发现仓库卫生问题。")
        return "\n".join(lines)
    for issue in report.issues:
        path = f" ({issue.path})" if issue.path else ""
        suggestion = f" 建议：{issue.suggestion}" if issue.suggestion else ""
        lines.append(f"- [{issue.severity}] {issue.code}{path}：{issue.message}{suggestion}")
    return "\n".join(lines)


def _run_release_commands() -> int:
    commands = [
        ["python", "-m", "ruff", "check", "."],
        ["python", "-m", "pytest"],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def _argv_from_args(args: argparse.Namespace, names: list[str]) -> list[str]:
    argv: list[str] = []
    for name in names:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
