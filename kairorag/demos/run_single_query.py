"""Run one KairoRAG query from the command line."""

from __future__ import annotations

import argparse
import json

from kairorag.agent.react_agent import ReactAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single KairoRAG query")
    parser.add_argument("--question", required=True)
    parser.add_argument("--verify-jobs", action="store_true")
    parser.add_argument("--apply-kb-updates", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-chunks-to-read", type=int, default=5)
    args = parser.parse_args()

    agent = ReactAgent(
        max_tool_calls=args.max_tool_calls,
        max_chunks_to_read=args.max_chunks_to_read,
    )
    result = agent.run(
        args.question,
        verify_jobs=args.verify_jobs,
        apply_kb_updates=args.apply_kb_updates,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation['chunk_id']}: {citation['title']}")
        print("\nRetrieval trace:")
        for event in result["retrieval_trace"]:
            print(f"- step {event['step']}: {event['tool']}")


if __name__ == "__main__":
    main()
