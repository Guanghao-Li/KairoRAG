"""使用 LLM tool calling 的 Agent planner。"""

from __future__ import annotations

import json
from typing import Any

from kairorag.cloud.agent.prompts import AGENT_SYSTEM_PROMPT
from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentState, AgentTraceStep
from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import LLMMessage, LLMProvider, LLMToolCall


class LLMToolPlanner:
    """让 LLM 基于当前状态选择下一组工具。"""

    def __init__(self, settings: KairoCloudSettings, llm_provider: LLMProvider, registry: ToolRegistry) -> None:
        self.settings = settings
        self.llm_provider = llm_provider
        self.registry = registry

    def next_tool_calls(self, state: AgentState) -> list[LLMToolCall]:
        """返回下一轮工具调用；必要时自动修复空工具调用。"""

        state.metrics["llm_planning_rounds"] = int(state.metrics.get("llm_planning_rounds", 0)) + 1
        messages = [
            LLMMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=_state_prompt(state, self.settings.max_tool_calls)),
        ]
        try:
            response = self.llm_provider.complete_with_tools(
                messages,
                self.registry.specs(),
                temperature=0.0,
            )
        except KairoProviderError as exc:
            state.trace.append(AgentTraceStep("planner_error", {"error": str(exc)}))
            return [LLMToolCall(id="planner-error", name="__invalid_tool_arguments__", arguments={"error": str(exc)})]

        state.trace.append(
            AgentTraceStep(
                "llm_planning",
                {
                    "round": state.metrics["llm_planning_rounds"],
                    "summary": _short_text(response.content),
                    "tool_names": [tool_call.name for tool_call in response.tool_calls],
                },
            )
        )
        tool_calls = [_normalize_tool_call(tool_call) for tool_call in response.tool_calls]
        if not tool_calls:
            repaired = _auto_repair_tool_call(state)
            state.trace.append(
                AgentTraceStep(
                    "planner_auto_repair",
                    {"tool_name": repaired.name, "reason": "LLM 未返回 tool call"},
                )
            )
            return [repaired]
        return tool_calls


def _state_prompt(state: AgentState, max_tool_calls: int) -> str:
    summary = {
        "query": state.query,
        "searched_chunk_ids": list(state.search_results.keys()),
        "read_chunk_ids": list(state.read_chunks.keys()),
        "freshness_results": [
            {
                "key": key,
                "status": result.status,
                "confidence": result.confidence,
                "recommended_action": result.recommended_action,
            }
            for key, result in state.freshness_results.items()
        ],
        "budget": {
            "max_tool_calls": max_tool_calls,
            "used_tool_calls": state.tool_call_count,
            "remaining_tool_calls": max(0, max_tool_calls - state.tool_call_count),
        },
        "filters": state.filters or {},
        "注意": "不要把 search 或 web_search 摘要当最终证据；freshness 问题必须先 verify_job_freshness，最终回答前必须 chunk_read。",
    }
    return "请根据当前状态选择下一步工具：\n" + json.dumps(summary, ensure_ascii=False, indent=2)


def _normalize_tool_call(tool_call: LLMToolCall) -> LLMToolCall:
    if not isinstance(tool_call.arguments, dict):
        return LLMToolCall(
            id=tool_call.id,
            name="__invalid_tool_arguments__",
            arguments={"tool": tool_call.name, "error": "tool arguments 必须是 JSON object。"},
        )
    return tool_call


def _auto_repair_tool_call(state: AgentState) -> LLMToolCall:
    if _is_freshness_query(state.query) and state.read_chunks and not state.freshness_results:
        return LLMToolCall(
            id="auto-verify-freshness",
            name="verify_job_freshness",
            arguments={"chunk_id": next(iter(state.read_chunks.keys()))},
        )
    if state.read_chunks:
        return LLMToolCall(id="auto-generate", name="generate_grounded_answer", arguments={"query": state.query})
    return LLMToolCall(
        id="auto-hybrid",
        name="hybrid_search",
        arguments={"query": state.query},
    )


def _short_text(text: Any, limit: int = 160) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _is_freshness_query(query: str) -> bool:
    lowered = query.lower()
    markers = ["还在招", "现在还开放", "active", "closed", "still open", "职位关闭", "是否还能申请"]
    return any(marker in lowered or marker in query for marker in markers)
