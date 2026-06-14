"""Cloud LLM Agent 的工具注册与执行。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from kairorag.cloud.answering import SAFE_REFUSAL, CloudAnswer, GroundedCloudAnswerGenerator
from kairorag.cloud.freshness import CloudFreshnessUpdater, CloudJobFreshnessVerifier
from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.observability import TraceEvent, TraceLogger, now_iso
from kairorag.cloud.agent.schemas import AgentState, AgentToolOutput, AgentTraceStep
from kairorag.cloud.agent.tools import agent_tool_specs
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult, RetrievalResult, RetrievalTraceStep
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMToolCall, LLMToolSpec
from kairorag.providers.vectorstores import VectorStoreProvider
from kairorag.providers.websearch import WebSearchProvider


TEXT_PREVIEW_LIMIT = 240


class ToolRegistry:
    """执行 Agent 工具并维护状态。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        retriever: Any,
        answer_generator: GroundedCloudAnswerGenerator,
        *,
        web_search_provider: WebSearchProvider | None = None,
        freshness_verifier: CloudJobFreshnessVerifier | None = None,
        freshness_updater: CloudFreshnessUpdater | None = None,
        vector_store: VectorStoreProvider | None = None,
        freshness_apply_authorized: bool = False,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.web_search_provider = web_search_provider
        self.freshness_verifier = freshness_verifier
        self.freshness_updater = freshness_updater
        self.vector_store = vector_store
        self.freshness_apply_authorized = freshness_apply_authorized
        self.trace_logger = trace_logger or TraceLogger(
            settings.trace_log_path,
            enabled=settings.observability_enabled,
        )
        self._handlers: dict[str, Callable[[LLMToolCall, AgentState], AgentToolOutput]] = {
            "semantic_search": self._semantic_search,
            "keyword_search": self._keyword_search,
            "hybrid_search": self._hybrid_search,
            "chunk_read": self._chunk_read,
            "web_search": self._web_search,
            "verify_job_freshness": self._verify_job_freshness,
            "apply_freshness_update": self._apply_freshness_update,
            "generate_grounded_answer": self._generate_grounded_answer,
            "finish": self._finish,
        }

    def specs(self) -> list[LLMToolSpec]:
        specs = agent_tool_specs()
        if self.freshness_updater is None or self.vector_store is None:
            return [spec for spec in specs if spec.name != "apply_freshness_update"]
        return specs

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        """执行工具，并把摘要写入 trace。"""

        self._trace(state, "tool_call", {"name": tool_call.name, "arguments": _safe_arguments(tool_call.arguments)})
        self.trace_logger.log(
            TraceEvent(
                event_type="agent_tool_called",
                timestamp=now_iso(),
                run_id=str(state.metrics.get("run_id", "agent")),
                query=state.query,
                detail={"name": tool_call.name, "arguments": _safe_arguments(tool_call.arguments)},
            )
        )
        handler = self._handlers.get(tool_call.name)
        if handler is None:
            return self._invalid(state, tool_call.name, f"不存在的工具：{tool_call.name}")
        output = handler(tool_call, state)
        self._trace(
            state,
            "tool_result",
            {"name": output.name, "ok": output.ok, "result": output.result, "error": output.error},
        )
        self._sync_metrics(state)
        return output

    def _semantic_search(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        return self._run_search("semantic_search", self.retriever.semantic_search, tool_call, state)

    def _keyword_search(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        return self._run_search("keyword_search", self.retriever.keyword_search, tool_call, state)

    def _hybrid_search(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        return self._run_search("hybrid_search", self.retriever.hybrid_search, tool_call, state)

    def _run_search(
        self,
        name: str,
        search_fn: Callable[..., list[CloudSearchResult]],
        tool_call: LLMToolCall,
        state: AgentState,
    ) -> AgentToolOutput:
        query = str(tool_call.arguments.get("query") or state.query).strip()
        if not query:
            return self._invalid(state, name, "search 工具缺少 query。")
        top_k = int(tool_call.arguments.get("top_k") or self.settings.max_search_results)
        filters = _merge_filters(state.filters, tool_call.arguments.get("filters"))
        results = search_fn(query, top_k=top_k, filters=filters)
        for result in results:
            state.search_results[result.chunk_id] = result
        return AgentToolOutput(
            name=name,
            ok=True,
            result={
                "chunk_ids": [result.chunk_id for result in results],
                "results": [_search_summary(result) for result in results],
                "注意": "search 摘要只是候选线索，不能作为最终答案证据。",
            },
        )

    def _chunk_read(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        raw_ids = tool_call.arguments.get("chunk_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            return self._invalid(state, "chunk_read", "chunk_read 需要非空 chunk_ids 列表。")
        chunk_ids = [str(chunk_id) for chunk_id in raw_ids]
        unseen = [chunk_id for chunk_id in chunk_ids if chunk_id not in state.search_results]
        if unseen:
            return self._invalid(state, "chunk_read", f"拒绝读取未 search 到的 chunk_id：{unseen}")
        chunks = self.retriever.chunk_read(chunk_ids)
        for chunk in chunks:
            state.read_chunks[chunk.chunk_id] = chunk
        return AgentToolOutput(
            name="chunk_read",
            ok=True,
            result={
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "chunks": [_read_summary(chunk) for chunk in chunks],
            },
        )

    def _web_search(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        if not self.settings.job_freshness_enabled or self.web_search_provider is None:
            return self._invalid(state, "web_search", "岗位 freshness 未启用，web_search 工具不可用。")
        query = str(tool_call.arguments.get("query") or "").strip()
        if not query:
            return self._invalid(state, "web_search", "web_search 需要 query。")
        max_results = int(tool_call.arguments.get("max_results") or self.settings.web_search_max_results)
        if max_results <= 0:
            return self._invalid(state, "web_search", "web_search max_results 必须大于 0。")
        results = self.web_search_provider.search(query, max_results=max_results)
        return AgentToolOutput(
            name="web_search",
            ok=True,
            result={
                "query": query,
                "result_count": len(results),
                "results": [_web_search_summary(result) for result in results],
                "注意": "web_search 结果只是网页线索，不能作为最终 RAG citation。",
            },
        )

    def _verify_job_freshness(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        if not self.settings.job_freshness_enabled or self.freshness_verifier is None:
            return self._invalid(state, "verify_job_freshness", "岗位 freshness verification 未启用。")
        chunk_id = str(tool_call.arguments.get("chunk_id") or "").strip() or None
        if chunk_id:
            chunk = state.read_chunks.get(chunk_id)
            if chunk is None:
                return self._invalid(
                    state,
                    "verify_job_freshness",
                    f"chunk_id={chunk_id} 尚未通过 chunk_read 读取，不能直接验证。",
                )
            result = self.freshness_verifier.verify_from_chunk(chunk)
        else:
            query = str(tool_call.arguments.get("query") or state.query).strip()
            company = _optional_arg(tool_call.arguments, "company")
            title = _optional_arg(tool_call.arguments, "title")
            job_id = _optional_arg(tool_call.arguments, "job_id")
            original_url = _optional_arg(tool_call.arguments, "original_url")
            if not any([query, company, title, job_id, original_url]):
                return self._invalid(
                    state,
                    "verify_job_freshness",
                    "verify_job_freshness 需要 chunk_id，或 company/title/job_id/original_url/query 中至少一项。",
                )
            result = self.freshness_verifier.verify_by_query(
                query,
                company=company,
                title=title,
                job_id=job_id,
                original_url=original_url,
            )

        key = result.chunk_id or result.job_id or result.original_url or f"query:{len(state.freshness_results) + 1}"
        state.freshness_results[key] = result
        state.metrics["freshness_verification_count"] = len(state.freshness_results)
        return AgentToolOutput(
            name="verify_job_freshness",
            ok=True,
            result=_freshness_summary(result),
        )

    def _apply_freshness_update(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        if self.freshness_updater is None or self.vector_store is None:
            return self._invalid(state, "apply_freshness_update", "freshness metadata 写回工具未启用。")
        chunk_id = str(tool_call.arguments.get("chunk_id") or "").strip()
        if not chunk_id:
            return self._invalid(state, "apply_freshness_update", "apply_freshness_update 需要 chunk_id。")
        result = _find_freshness_result(state.freshness_results, chunk_id)
        if result is None:
            return self._invalid(
                state,
                "apply_freshness_update",
                f"chunk_id={chunk_id} 尚无 freshness verification 结果，不能写回。",
            )
        requested_dry_run = _bool_arg(
            tool_call.arguments.get("dry_run"),
            default=self.settings.qdrant_metadata_write_dry_run,
        )
        authorized = self.freshness_apply_authorized or _query_authorizes_freshness_apply(state.query)
        forced_dry_run = bool(requested_dry_run or not authorized)
        update_result = self.freshness_updater.apply_update(
            result,
            vector_store=self.vector_store,
            dry_run=forced_dry_run,
            user_confirmed=False,
            agent_initiated=True,
        )
        state.metrics["freshness_update_count"] = int(state.metrics.get("freshness_update_count", 0)) + 1
        state.metrics["freshness_update_applied_count"] = int(state.metrics.get("freshness_update_applied_count", 0)) + int(
            update_result.applied
        )
        self._trace(
            state,
            "freshness_update",
            {
                "chunk_id": chunk_id,
                "requested_dry_run": requested_dry_run,
                "effective_dry_run": forced_dry_run,
                "authorized": authorized,
                "applied": update_result.applied,
                "audit_log_path": update_result.audit_log_path,
                "error": update_result.error,
                "approval_decision": update_result.approval_decision.to_dict()
                if update_result.approval_decision
                else None,
            },
        )
        output = {
            "plan": update_result.plan.to_dict(),
            "applied": update_result.applied,
            "audit_log_path": update_result.audit_log_path,
            "effective_dry_run": update_result.plan.dry_run,
            "approval_decision": update_result.approval_decision.to_dict()
            if update_result.approval_decision
            else None,
        }
        if not authorized and requested_dry_run is False:
            output["guardrail"] = "用户未明确授权写回，已强制 dry-run。"
        if update_result.error:
            return AgentToolOutput(
                name="apply_freshness_update",
                ok=False,
                result=output,
                error=update_result.error,
            )
        return AgentToolOutput(name="apply_freshness_update", ok=True, result=output)

    def _generate_grounded_answer(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        if not state.read_chunks:
            state.blocked_final_answer_count += 1
            self._trace(state, "blocked_tool_call", {"name": "generate_grounded_answer", "reason": "没有 read_chunks"})
            return AgentToolOutput(
                name="generate_grounded_answer",
                ok=False,
                result={},
                error="没有 read_chunks，必须先调用 chunk_read。",
            )
        if _is_freshness_query(state.query) and not state.freshness_results and self.settings.job_freshness_enabled:
            state.blocked_final_answer_count += 1
            self._trace(
                state,
                "blocked_tool_call",
                {"name": "generate_grounded_answer", "reason": "freshness query 缺少 verification result"},
            )
            return AgentToolOutput(
                name="generate_grounded_answer",
                ok=False,
                result={},
                error="这是岗位实时状态问题，必须先调用 verify_job_freshness。",
            )
        query = str(tool_call.arguments.get("query") or state.query)
        retrieval_result = RetrievalResult(
            query=query,
            search_results=list(state.search_results.values()),
            read_chunks=list(state.read_chunks.values()),
            trace=[RetrievalTraceStep(step=step.step, detail=dict(step.detail)) for step in state.trace],
            metrics=dict(state.metrics),
        )
        answer = _call_answer_generator(
            self.answer_generator,
            query,
            retrieval_result,
            freshness_results=state.freshness_results,
        )
        final_answer = _apply_freshness_guardrail(state.query, answer, state.freshness_results)
        state.final_answer = final_answer.answer
        state.final_citations = final_answer.citations
        state.finished = True
        state.metrics.update({f"answer_{key}": value for key, value in final_answer.metrics.items()})
        state.metrics["final_citation_count"] = len(final_answer.citations)
        state.metrics["freshness_verification_count"] = len(state.freshness_results)
        if final_answer.metrics.get("freshness_guardrail_applied"):
            state.metrics["freshness_guardrail_applied"] = True
        return AgentToolOutput(
            name="generate_grounded_answer",
            ok=True,
            result={
                "answer": state.final_answer,
                "citation_count": len(state.final_citations),
                "citations": [asdict(citation) for citation in state.final_citations],
                "verification_results": [_freshness_summary(result) for result in state.freshness_results.values()],
            },
        )

    def _finish(self, tool_call: LLMToolCall, state: AgentState) -> AgentToolOutput:
        reason = str(tool_call.arguments.get("reason") or "")
        if not state.final_answer:
            if _is_freshness_query(state.query) and not state.freshness_results:
                state.final_answer = "本次没有岗位 freshness verification 结果，不能确认实时状态。"
                state.metrics["freshness_guardrail_applied"] = True
            else:
                state.final_answer = SAFE_REFUSAL
            state.final_citations = []
        state.finished = True
        return AgentToolOutput(
            name="finish",
            ok=True,
            result={"answer": state.final_answer, "reason": reason, "citation_count": len(state.final_citations)},
        )

    def _invalid(self, state: AgentState, name: str, error: str) -> AgentToolOutput:
        state.invalid_tool_call_count += 1
        self._trace(state, "illegal_tool_call", {"name": name, "error": error})
        return AgentToolOutput(name=name, ok=False, result={}, error=error)

    def _trace(self, state: AgentState, step: str, detail: dict[str, Any]) -> None:
        state.trace.append(AgentTraceStep(step=step, detail=_sanitize_detail(detail)))

    def _sync_metrics(self, state: AgentState) -> None:
        state.metrics["tool_call_count"] = state.tool_call_count
        state.metrics["invalid_tool_call_count"] = state.invalid_tool_call_count
        state.metrics["blocked_final_answer_count"] = state.blocked_final_answer_count
        state.metrics["read_chunk_count"] = len(state.read_chunks)
        state.metrics["search_result_count"] = len(state.search_results)
        state.metrics["final_citation_count"] = len(state.final_citations)
        state.metrics["freshness_verification_count"] = len(state.freshness_results)


def _search_summary(result: CloudSearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "score": result.score,
        "source": result.source,
        "metadata": dict(result.metadata),
        "text_preview": _preview(result.text),
        "可作为最终证据": False,
    }


def _read_summary(chunk: CloudReadChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "metadata": dict(chunk.metadata),
        "text_preview": _preview(chunk.text),
    }


def _web_search_summary(result: Any) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": _preview(result.snippet),
        "source": result.source,
        "published_at": result.published_at,
        "可作为最终证据": False,
    }


def _freshness_summary(result: JobFreshnessResult) -> dict[str, Any]:
    return {
        "job_id": result.job_id,
        "chunk_id": result.chunk_id,
        "company": result.company,
        "title": result.title,
        "original_url": result.original_url,
        "status": result.status,
        "confidence": result.confidence,
        "recommended_action": result.recommended_action,
        "reason": result.reason,
        "new_url": result.new_url,
        "evidence": [
            {
                "source": evidence.source,
                "url": evidence.url,
                "title": evidence.title,
                "signal": evidence.signal,
                "confidence": evidence.confidence,
                "snippet": _preview(evidence.snippet),
                "observed_at": evidence.observed_at,
            }
            for evidence in result.evidence
        ],
    }


def _optional_arg(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _call_answer_generator(
    answer_generator: GroundedCloudAnswerGenerator,
    query: str,
    retrieval_result: RetrievalResult,
    *,
    freshness_results: dict[str, JobFreshnessResult],
) -> CloudAnswer:
    try:
        return answer_generator.answer(query, retrieval_result, freshness_results=freshness_results)
    except TypeError as exc:
        if "freshness_results" not in str(exc):
            raise
        return answer_generator.answer(query, retrieval_result)


def _merge_filters(default_filters: dict[str, Any] | None, requested_filters: Any) -> dict[str, Any] | None:
    merged = dict(default_filters or {})
    if isinstance(requested_filters, dict):
        merged.update({key: value for key, value in requested_filters.items() if value is not None})
    return merged or None


def _safe_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"非法参数": str(arguments)}
    safe = {
        key: value
        for key, value in dict(arguments).items()
        if not any(marker in str(key).lower() for marker in ("key", "token", "secret"))
    }
    if "text" in safe:
        safe["text"] = _preview(str(safe["text"]))
    return safe


def _find_freshness_result(results: dict[str, JobFreshnessResult], chunk_id: str) -> JobFreshnessResult | None:
    direct = results.get(chunk_id)
    if direct is not None:
        return direct
    for result in results.values():
        if result.chunk_id == chunk_id:
            return result
    return None


def _bool_arg(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "apply", "写回"}
    return bool(value)


def _query_authorizes_freshness_apply(query: str) -> bool:
    lowered = query.lower()
    markers = [
        "写回",
        "apply",
        "更新索引",
        "标记关闭",
        "标记为 archived",
        "标记为archive",
        "archive",
        "归档",
    ]
    return any(marker in lowered or marker in query for marker in markers)


def _sanitize_detail(detail: dict[str, Any]) -> dict[str, Any]:
    text = str(detail)
    for sensitive in ("OPENAI_API_KEY", "QDRANT_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY", "BING_API_KEY"):
        if sensitive in text:
            return {"已隐藏": "trace detail 包含敏感字段名，已省略。"}
    return detail


def _preview(text: str, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _is_freshness_query(query: str) -> bool:
    lowered = query.lower()
    markers = [
        "还在招",
        "active",
        "closed",
        "现在还开放",
        "仍在招聘",
        "还开放",
        "是否开放",
        "还有效",
        "实时状态",
        "still open",
        "no longer available",
        "职位关闭",
        "是否还能申请",
    ]
    return any(marker in lowered or marker in query for marker in markers)


def _has_active_claim(answer: str) -> bool:
    lowered = answer.lower()
    markers = [
        "还在招",
        "仍在招聘",
        "正在招聘",
        "仍 active",
        "is active",
        "still active",
        "currently open",
        "还开放",
        "可申请",
    ]
    return any(marker in lowered or marker in answer for marker in markers)


def _apply_freshness_guardrail(
    query: str,
    answer: CloudAnswer,
    freshness_results: dict[str, JobFreshnessResult],
) -> CloudAnswer:
    if not _is_freshness_query(query) and not freshness_results:
        return answer
    statuses = {result.status for result in freshness_results.values()}
    if not freshness_results:
        prefix = "本次没有岗位 freshness verification 结果，不能确认实时状态。"
    elif "closed" in statuses:
        prefix = "本次 verification 判定岗位已关闭，不能作为当前可申请岗位推荐。"
    elif statuses & {"stale", "unknown"}:
        prefix = "本次 verification 不能确认岗位实时状态，需要人工确认或更多证据。"
    elif "active" in statuses:
        prefix = "基于本次 web verification evidence，岗位倾向于 active。"
    elif "updated" in statuses:
        prefix = "本次 verification 找到可能更新后的岗位链接，请优先核对新 URL。"
    elif "duplicate" in statuses:
        prefix = "本次 verification 判定该岗位为重复记录，不应重复推荐。"
    else:
        prefix = "本次岗位 verification 已完成。"
    if answer.answer.startswith(prefix):
        guarded = answer.answer
    elif "closed" in statuses and _has_active_claim(answer.answer):
        guarded = prefix + "原回答中涉及可申请表述，已按 verification 结果收紧。"
    else:
        guarded = prefix + answer.answer
    metrics = {**answer.metrics, "freshness_guardrail_applied": True}
    return CloudAnswer(
        answer=guarded,
        citations=answer.citations,
        retrieval_trace=answer.retrieval_trace,
        metrics=metrics,
    )
