"""Cloud LLM Agent 可调用工具定义。"""

from __future__ import annotations

from kairorag.providers.llm import LLMToolSpec


def agent_tool_specs() -> list[LLMToolSpec]:
    """返回 planner 可见的工具 schema。"""

    query_search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
            "filters": {"type": "object"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return [
        LLMToolSpec(
            name="semantic_search",
            description="使用 OpenAI query embedding 与 Qdrant 做语义检索，只返回候选线索。",
            parameters=query_search_schema,
        ),
        LLMToolSpec(
            name="keyword_search",
            description="使用 BM25 做关键词检索，只返回候选线索。",
            parameters=query_search_schema,
        ),
        LLMToolSpec(
            name="hybrid_search",
            description="使用 RRF 融合 semantic_search 和 keyword_search，只返回候选线索。",
            parameters=query_search_schema,
        ),
        LLMToolSpec(
            name="chunk_read",
            description="读取已经检索到的 chunk，完整 chunk 才能作为最终回答证据。",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["chunk_ids"],
                "additionalProperties": False,
            },
        ),
        LLMToolSpec(
            name="web_search",
            description="调用真实 Web Search provider 获取网页线索；结果不能作为最终 RAG citation。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        LLMToolSpec(
            name="verify_job_freshness",
            description="验证岗位是否 active、closed、stale、updated、duplicate 或 unknown。",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "job_id": {"type": "string"},
                    "original_url": {"type": "string"},
                    "query": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
        LLMToolSpec(
            name="apply_freshness_update",
            description="根据已有 freshness verification 结果生成或安全执行 metadata 写回；默认 dry-run。",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        ),
        LLMToolSpec(
            name="generate_grounded_answer",
            description="基于已 read chunks 调用 grounded answer generator 生成最终答案。",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        LLMToolSpec(
            name="finish",
            description="结束循环。没有最终答案时只能安全拒答，不能直接写自然语言答案。",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
    ]
