"""Run batch KairoRAG evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kairorag.config import RESULTS_DIR
from kairorag.evaluation.evaluator import run_job_verification_eval, run_qa_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KairoRAG batch evaluation")
    parser.add_argument("--eval-file", default="data/eval/qa_eval.json")
    parser.add_argument("--output", default=str(RESULTS_DIR / "eval_report.json"))
    parser.add_argument("--job-verification", action="store_true")
    parser.add_argument("--mock-web", action="store_true")
    parser.add_argument("--include-traces", action="store_true")
    args = parser.parse_args()

    if args.job_verification:
        payload = run_job_verification_eval(args.eval_file, args.output)
    else:
        payload = run_qa_eval(args.eval_file, args.output, include_traces=args.include_traces)
    print(json.dumps(payload["overall_metrics"], ensure_ascii=False, indent=2))
    print(f"Report written to {Path(args.output)}")


if __name__ == "__main__":
    main()

