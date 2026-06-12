"""基于 read chunks 的 grounded answer 生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk, RetrievalResult, RetrievalTraceStep
from kairorag.config import KairoCloudSettings
from kairorag.ingestion.metadata import estimate_tokens
from kairorag.providers.llm import LLMMessage, LLMProvider


SAFE_REFUSAL = "知识库证据不足，无法可靠回答该问题。"


@dataclass(frozen=True)
class CloudCitation:
    chunk_id: str
    doc_id: str
    title: str
    evidence: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CloudAnswer:
    answer: str
    citations: list[CloudCitation]
    retrieval_trace: list[RetrievalTraceStep]
    metrics: dict[str, Any]
    verification_results: list[JobFreshnessResult] = field(default_factory=list)


class GroundedCloudAnswerGenerator:
    """只允许基于已读取 chunk 生成最终回答。"""

    def __init__(self, settings: KairoCloudSettings, llm_provider: LLMProvider) -> None:
        self.settings = settings
        self.llm_provider = llm_provider

    def answer(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        *,
        freshness_results: dict[str, JobFreshnessResult] | None = None,
    ) -> CloudAnswer:
        """调用 LLM 生成 JSON 回答，并校验 citation 只来自 read chunks。"""

        verification_results = list((freshness_results or {}).values())
        if not retrieval_result.read_chunks:
            return CloudAnswer(
                answer=SAFE_REFUSAL,
                citations=[],
                retrieval_trace=retrieval_result.trace,
                metrics=self._metrics(
                    retrieval_result.read_chunks,
                    [],
                    0,
                    llm_model=self._llm_model(),
                    verification_results=verification_results,
                ),
                verification_results=verification_results,
            )

        messages = [
            LLMMessage(role="system", content=_system_prompt()),
            LLMMessage(
                role="user",
                content=_user_prompt(query, retrieval_result.read_chunks, verification_results),
            ),
        ]
        response = self.llm_provider.complete(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            payload = _parse_json_object(response.content)
        except ValueError:
            return CloudAnswer(
                answer=SAFE_REFUSAL,
                citations=[],
                retrieval_trace=retrieval_result.trace,
                metrics={
                    **self._metrics(
                        retrieval_result.read_chunks,
                        [],
                        0,
                        llm_model=response.model,
                        verification_results=verification_results,
                    ),
                    "invalid_json": True,
                },
                verification_results=verification_results,
            )

        raw_answer = str(payload.get("answer", "")).strip() or SAFE_REFUSAL
        citations, invalid_count = _validate_citations(payload.get("citations", []), retrieval_result.read_chunks)
        raw_answer = _apply_answer_freshness_guardrail(raw_answer, verification_results)
        if _contains_forbidden_snippet_phrase(raw_answer):
            raw_answer = SAFE_REFUSAL
            citations = []
            invalid_count += 1
        return CloudAnswer(
            answer=raw_answer,
            citations=citations,
            retrieval_trace=retrieval_result.trace,
            metrics=self._metrics(
                retrieval_result.read_chunks,
                citations,
                invalid_count,
                llm_model=response.model,
                verification_results=verification_results,
            ),
            verification_results=verification_results,
        )

    def _metrics(
        self,
        read_chunks: list[CloudReadChunk],
        citations: list[CloudCitation],
        invalid_citation_count: int,
        *,
        llm_model: str,
        verification_results: list[JobFreshnessResult] | None = None,
    ) -> dict[str, Any]:
        return {
            "read_chunk_count": len(read_chunks),
            "citation_count": len(citations),
            "invalid_citation_count": invalid_citation_count,
            "llm_model": llm_model,
            "context_token_estimate": sum(estimate_tokens(chunk.text) for chunk in read_chunks),
            "verification_result_count": len(verification_results or []),
        }

    def _llm_model(self) -> str:
        return str(getattr(self.llm_provider, "model", self.settings.openai_chat_model))


def _system_prompt() -> str:
    return (
        "你是 KairoRAG 的 grounded answer 生成器。只能使用用户提供的 chunks 回答，"
        "不要使用外部知识。知识库证据不足时，直接回答“知识库证据不足，无法可靠回答该问题”。"
        "每个关键结论必须给出 chunk_id citation，不允许引用未提供的 chunk_id。"
        "如果提供了 freshness verification summary，必须遵守其中的状态约束：closed 不得说当前可申请；"
        "stale/unknown 必须说明实时状态未确认；active 只能表述为“基于 web verification evidence 倾向 active”。"
        "freshness evidence 不是 chunk citation，不得把网页 snippet 写进 citations。"
        "输出必须是 JSON object，格式为："
        '{"answer":"string","citations":[{"chunk_id":"string","evidence":"string"}]}。'
    )


def _user_prompt(
    query: str,
    read_chunks: list[CloudReadChunk],
    verification_results: list[JobFreshnessResult],
) -> str:
    chunk_blocks = []
    for chunk in read_chunks:
        chunk_blocks.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.chunk_id}",
                    f"doc_id: {chunk.doc_id}",
                    f"title: {chunk.title}",
                    "text:",
                    chunk.text,
                ]
            )
        )
    prompt = "问题：\n" + query + "\n\n可用 chunks：\n" + "\n\n---\n\n".join(chunk_blocks)
    if verification_results:
        prompt += "\n\nfreshness verification summary：\n" + json.dumps(
            [_verification_summary(result) for result in verification_results],
            ensure_ascii=False,
            indent=2,
        )
    return prompt


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM 未返回合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON 输出必须是 object。")
    return payload


def _verification_summary(result: JobFreshnessResult) -> dict[str, Any]:
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
        "verification_evidence": [
            {
                "source": evidence.source,
                "url": evidence.url,
                "title": evidence.title,
                "signal": evidence.signal,
                "confidence": evidence.confidence,
                "snippet": evidence.snippet,
            }
            for evidence in result.evidence
        ],
    }


def _apply_answer_freshness_guardrail(answer: str, verification_results: list[JobFreshnessResult]) -> str:
    if not verification_results:
        return answer
    statuses = {result.status for result in verification_results}
    if "closed" in statuses and _has_active_or_apply_claim(answer):
        return "本次 verification 判定岗位已关闭，不能作为当前可申请岗位推荐。"
    if statuses & {"stale", "unknown"} and "未确认" not in answer:
        return "本次 verification 不能确认岗位实时状态，需要人工确认或更多证据。" + answer
    if "active" in statuses and "web verification evidence" not in answer:
        return "基于本次 web verification evidence，岗位倾向于 active。" + answer
    return answer


def _has_active_or_apply_claim(answer: str) -> bool:
    lowered = answer.lower()
    markers = ["可申请", "还在招", "仍在招聘", "正在招聘", "active", "still open", "currently open"]
    return any(marker in lowered or marker in answer for marker in markers)


def _validate_citations(
    raw_citations: Any,
    read_chunks: list[CloudReadChunk],
) -> tuple[list[CloudCitation], int]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in read_chunks}
    valid: list[CloudCitation] = []
    invalid_count = 0
    if not isinstance(raw_citations, list):
        return [], 1
    for item in raw_citations:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None or not evidence or evidence not in chunk.text:
            invalid_count += 1
            continue
        valid.append(
            CloudCitation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                evidence=evidence,
                metadata=dict(chunk.metadata),
            )
        )
    return valid, invalid_count


def _contains_forbidden_snippet_phrase(answer: str) -> bool:
    lowered = answer.lower()
    return "搜索结果 snippet" in answer or "search result snippet" in lowered
