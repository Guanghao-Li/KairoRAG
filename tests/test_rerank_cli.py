from kairorag.cloud.agent.cli import build_parser as build_agent_parser
from kairorag.cloud.query_cli import build_parser as build_query_parser


def test_query_cli_parses_rerank_and_writeback_flags():
    args = build_query_parser().parse_args(
        [
            "哪些岗位要求 RAG？",
            "--reranker",
            "cohere",
            "--rerank-top-k",
            "3",
            "--apply-freshness-update",
            "--dry-run",
            "false",
            "--audit-log",
            "data/audit.jsonl",
            "--trace-output",
            "data/run_trace.json",
        ]
    )

    assert args.reranker == "cohere"
    assert args.rerank_top_k == 3
    assert args.apply_freshness_update is True
    assert args.dry_run == "false"
    assert args.audit_log == "data/audit.jsonl"
    assert args.trace_output == "data/run_trace.json"


def test_agent_cli_parses_rerank_and_writeback_flags():
    args = build_agent_parser().parse_args(
        [
            "把关闭岗位标记为 archived",
            "--reranker",
            "openai_listwise",
            "--rerank-top-k",
            "2",
            "--apply-freshness-update",
            "--dry-run",
            "false",
        ]
    )

    assert args.reranker == "openai_listwise"
    assert args.rerank_top_k == 2
    assert args.apply_freshness_update is True
    assert args.dry_run == "false"
