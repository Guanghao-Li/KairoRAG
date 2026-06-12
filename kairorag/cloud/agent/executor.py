"""Cloud LLM Agent 执行循环。"""

from __future__ import annotations

from uuid import uuid4

from kairorag.cloud.answering import SAFE_REFUSAL
from kairorag.cloud.observability import TraceEvent, now_iso
from kairorag.cloud.agent.planner import LLMToolPlanner
from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentRunResult, AgentState, AgentTraceStep
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall


class LLMAgentExecutor:
    """执行 LLM planning -> tool call -> final answer 的循环。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        planner: LLMToolPlanner,
        registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.planner = planner
        self.registry = registry

    def run(self, query: str, *, filters: dict[str, object] | None = None) -> AgentRunResult:
        if not query or not query.strip():
            raise ValueError("query 不能为空。")
        state = AgentState(query=query.strip(), filters=filters)
        state.metrics["run_id"] = f"agent-{uuid4()}"
        state.trace.append(AgentTraceStep("agent_start", {"query": state.query, "filters": filters or {}}))

        while not state.finished and state.tool_call_count < self.settings.max_tool_calls:
            tool_calls = self.planner.next_tool_calls(state)
            for tool_call in tool_calls:
                if state.finished:
                    break
                if state.tool_call_count >= self.settings.max_tool_calls:
                    break
                state.tool_call_count += 1
                self.registry.execute(tool_call, state)

        if not state.finished:
            state.metrics["max_tool_calls_reached"] = True
            state.trace.append(AgentTraceStep("max_tool_calls_reached", {"max_tool_calls": self.settings.max_tool_calls}))
            if state.read_chunks:
                self.registry.execute(
                    LLMToolCall(
                        id="forced-final-answer",
                        name="generate_grounded_answer",
                        arguments={"query": state.query},
                    ),
                    state,
                )
            else:
                state.final_answer = SAFE_REFUSAL
                state.final_citations = []
                state.finished = True
                state.trace.append(AgentTraceStep("safe_refusal", {"reason": "工具预算耗尽且没有 read_chunks"}))

        _sync_final_metrics(state)
        self.registry.trace_logger.log(
            TraceEvent(
                event_type="agent_finished",
                timestamp=now_iso(),
                run_id=str(state.metrics.get("run_id", "agent")),
                query=state.query,
                detail={
                    "tool_call_count": state.metrics["tool_call_count"],
                    "invalid_tool_call_count": state.metrics["invalid_tool_call_count"],
                    "final_citation_count": state.metrics["final_citation_count"],
                },
            )
        )
        return AgentRunResult(
            query=state.query,
            answer=state.final_answer or SAFE_REFUSAL,
            citations=list(state.final_citations),
            trace=list(state.trace),
            metrics=dict(state.metrics),
            verification_results=list(state.freshness_results.values()),
        )


def _sync_final_metrics(state: AgentState) -> None:
    state.metrics["tool_call_count"] = state.tool_call_count
    state.metrics["invalid_tool_call_count"] = state.invalid_tool_call_count
    state.metrics["blocked_final_answer_count"] = state.blocked_final_answer_count
    state.metrics["read_chunk_count"] = len(state.read_chunks)
    state.metrics["search_result_count"] = len(state.search_results)
    state.metrics["final_citation_count"] = len(state.final_citations)
    state.metrics["freshness_verification_count"] = len(state.freshness_results)
