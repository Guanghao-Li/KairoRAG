"""Legacy / Deprecated：确定性答案生成器，仅保留给离线 demo 和评测对照。"""

from __future__ import annotations

from typing import Any

from kairorag.indexing.embedding import tokenize
from kairorag.schemas import Citation, JobVerificationResult, ReadChunkResult


def _evidence_text(blocks: list[dict[str, str]], chunk: ReadChunkResult) -> str:
    for block in blocks:
        if block.get("chunk_id") == chunk.chunk_id:
            return block.get("text", "")
    return chunk.text[:260]


def _extract_skills(text: str) -> list[str]:
    known = [
        "Python",
        "RAG",
        "LangGraph",
        "Vector Search",
        "Evaluation",
        "MCP",
        "Tool Calling",
        "Multi-Agent",
        "Hybrid Search",
        "Groundedness",
        "Audit Logs",
        "Index Refresh",
    ]
    lowered = text.lower()
    return [skill for skill in known if skill.lower() in lowered]


def generate_answer(
    question: str,
    read_chunks: list[ReadChunkResult],
    compressed_context: list[dict[str, str]],
    retrieval_trace: list[dict[str, Any]],
    verification_results: list[JobVerificationResult] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy：只基于已读取 chunk 生成确定性答案。"""

    verification_results = verification_results or []
    metrics = metrics or {}
    valid_chunks = [chunk for chunk in read_chunks if chunk.text]
    if not valid_chunks:
        return {
            "answer": "现有知识库中没有足够证据。",
            "active_jobs": [],
            "closed_or_stale_jobs": [],
            "knowledge_base_updates": [],
            "citations": [],
            "retrieval_trace": retrieval_trace,
            "verification_trace": [result.to_dict() for result in verification_results],
            "metrics": metrics,
        }

    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            source_type=str(chunk.metadata.get("source_type", chunk.metadata.get("source_type", ""))),
            title=str(chunk.metadata.get("title", "")),
            evidence=_evidence_text(compressed_context, chunk),
        )
        for chunk in valid_chunks
    ]
    verified_by_job = {result.job_id: result for result in verification_results}
    active_jobs: list[dict[str, Any]] = []
    closed_or_stale: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    all_text = "\n".join(chunk.text for chunk in valid_chunks)

    for chunk in valid_chunks:
        metadata = chunk.metadata
        if metadata.get("source_type") != "job":
            continue
        job_id = str(metadata.get("job_id", metadata.get("doc_id", "")))
        verified = verified_by_job.get(job_id)
        status = verified.new_status if verified else str(metadata.get("verification_status", "unknown"))
        job_item = {
            "job_id": job_id,
            "company": metadata.get("company"),
            "title": metadata.get("title", "").split(" - ", 1)[-1],
            "verification_status": "active" if status == "updated" else status,
            "confidence": verified.confidence if verified else metadata.get("verification_confidence", 0.0),
            "last_verified_at": verified.verified_at if verified else metadata.get("last_verified_at", ""),
            "evidence_urls": [item.get("url") for item in verified.evidence if item.get("url")]
            if verified
            else metadata.get("evidence_urls", []),
            "skills": _extract_skills(chunk.text),
            "citation": chunk.chunk_id,
        }
        if status in {"active", "updated"}:
            active_jobs.append(job_item)
        elif status in {"closed", "stale", "unknown", "duplicate"}:
            closed_or_stale.append(job_item)
        if verified and verified.updated_fields:
            updates.append({"job_id": job_id, "updated_fields": verified.updated_fields})

    query_terms = set(tokenize(question))
    focused_lines = []
    for block in compressed_context:
        text = block.get("text", "")
        if query_terms & set(tokenize(text)) or not focused_lines:
            focused_lines.append(f"- {text} [{block.get('chunk_id')}]")
        if len(focused_lines) >= 5:
            break

    if active_jobs:
        job_lines = [
            f"- {job['company']} / {job['title']} ({job['verification_status']}, confidence={job['confidence']}) [{job['citation']}]"
            for job in active_jobs
        ]
        answer = "基于已读取 chunk，匹配到以下可优先关注的岗位：\n" + "\n".join(job_lines)
    else:
        answer = "基于已读取 chunk，相关证据如下：\n" + "\n".join(focused_lines)

    if closed_or_stale:
        answer += "\n\n不确定或不应默认推荐的岗位已单独列出，其中 stale 表示无法确认当前状态。"

    metrics.setdefault("predicted_skills", _extract_skills(all_text))
    return {
        "answer": answer,
        "active_jobs": active_jobs,
        "closed_or_stale_jobs": closed_or_stale,
        "knowledge_base_updates": updates,
        "citations": [citation.to_dict() for citation in citations],
        "retrieval_trace": retrieval_trace,
        "verification_trace": [result.to_dict() for result in verification_results],
        "metrics": metrics,
    }
