# KairoRAG RAG 评测报告

## 总体指标

| 指标 | 数值 |
| --- | ---: |
| context_token_usage | 412.400 |
| evidence_recall | 0.800 |
| groundedness_score | 1.000 |
| latency_ms | 2.200 |
| retrieval_hit_rate | 1.000 |
| skill_extraction_f1 | 0.262 |
| skill_extraction_precision | 0.169 |
| skill_extraction_recall | 0.600 |
| tool_call_count | 8.000 |

## 逐条样例

### qa_001
- 问题：哪些岗位要求 RAG、LangGraph 或多 Agent 编排经验？
- 命中的 gold chunks：['job_001_chunk_000', 'job_004_chunk_000']
- 未命中的 gold chunks：[]
- 检索路径：planner -> hybrid_search -> rerank_context -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> compress_context
- 指标明细：`{"retrieval_hit_rate": 1.0, "evidence_recall": 1.0, "groundedness_score": 1.0, "skill_extraction_precision": 0.3, "skill_extraction_recall": 1.0, "skill_extraction_f1": 0.4615384615384615, "context_token_usage": 418.0, "tool_call_count": 8.0, "latency_ms": 4.0}`

### qa_002
- 问题：哪些岗位更看重 RAG evaluation 和 groundedness？
- 命中的 gold chunks：['job_007_chunk_000']
- 未命中的 gold chunks：[]
- 检索路径：planner -> hybrid_search -> rerank_context -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> compress_context
- 指标明细：`{"retrieval_hit_rate": 1.0, "evidence_recall": 1.0, "groundedness_score": 1.0, "skill_extraction_precision": 0.1111111111111111, "skill_extraction_recall": 0.5, "skill_extraction_f1": 0.1818181818181818, "context_token_usage": 400.0, "tool_call_count": 8.0, "latency_ms": 2.0}`

### qa_003
- 问题：我的 akashic-agent 项目能匹配哪些 JD 要求？
- 命中的 gold chunks：['resume_chunk_000']
- 未命中的 gold chunks：['job_004_chunk_000']
- 检索路径：planner -> hybrid_search -> rerank_context -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> compress_context
- 指标明细：`{"retrieval_hit_rate": 1.0, "evidence_recall": 0.5, "groundedness_score": 1.0, "skill_extraction_precision": 0.25, "skill_extraction_recall": 1.0, "skill_extraction_f1": 0.4, "context_token_usage": 503.0, "tool_call_count": 8.0, "latency_ms": 2.0}`

### qa_004
- 问题：哪些公司资料强调企业知识库和 citation？
- 命中的 gold chunks：['company_civicflow_ai_chunk_000', 'company_northstar_ai_chunk_000']
- 未命中的 gold chunks：[]
- 检索路径：planner -> hybrid_search -> rerank_context -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> compress_context
- 指标明细：`{"retrieval_hit_rate": 1.0, "evidence_recall": 1.0, "groundedness_score": 1.0, "skill_extraction_precision": 0.0, "skill_extraction_recall": 0.0, "skill_extraction_f1": 0.0, "context_token_usage": 368.0, "tool_call_count": 8.0, "latency_ms": 2.0}`

### qa_005
- 问题：动态知识库维护需要解决哪些问题？
- 命中的 gold chunks：['job_008_chunk_000']
- 未命中的 gold chunks：['interview_dynamic_kb_chunk_000']
- 检索路径：planner -> semantic_search -> rerank_context -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> chunk_read -> compress_context
- 指标明细：`{"retrieval_hit_rate": 1.0, "evidence_recall": 0.5, "groundedness_score": 1.0, "skill_extraction_precision": 0.18181818181818182, "skill_extraction_recall": 0.5, "skill_extraction_f1": 0.26666666666666666, "context_token_usage": 373.0, "tool_call_count": 8.0, "latency_ms": 1.0}`
