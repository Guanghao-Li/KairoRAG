# KairoRAG：Agentic Knowledge Retrieval System

## 项目概述

KairoRAG 是一个面向企业知识库的 Agentic RAG 检索、评测与动态维护系统。它围绕 JD、简历、公司资料和面试材料构建，采用后端优先的实现方式，在不依赖付费 API 的前提下即可本地运行。

这个项目重点展示面向真实工程场景的 RAG 能力：文档加载、切分、关键词索引、语义索引、混合检索、chunk 级证据引用、上下文预算控制、检索轨迹记录、确定性 RAG 评测，以及岗位 freshness 验证与知识库维护。

## 为什么是 Agentic RAG

普通 top-k RAG：

```text
question -> vector search top-k -> 把 chunks 塞进 prompt -> answer
```

KairoRAG：

```text
question -> Agent 规划检索路径
  -> keyword_search / semantic_search / hybrid_search
  -> rerank
  -> chunk_read
  -> context budget
  -> 保留 citation 的上下文压缩
  -> 基于证据生成答案
  -> eval
  -> 需要时验证并更新知识库
```

关键区别在于：搜索结果里的 snippet 只是线索，不是最终证据。最终回答必须建立在 `chunk_read` 读到的内容，或保留原始 `chunk_id` 的压缩证据块之上。

## 核心特性

- 分层检索工具：提供 `keyword_search`、`semantic_search`、`hybrid_search` 和 `chunk_read`。
- Agentic 检索规划：planner 会根据精确术语、语义意图和 freshness 需求决定检索路径。
- 混合检索：通过 reciprocal rank fusion 组合关键词检索和本地向量检索。
- Chunk 级证据引用：每条回答引用都包含支持它的 `chunk_id`。
- 上下文预算控制：限制搜索结果数、读取 chunk 数、上下文 token 数和工具调用次数。
- 检索轨迹记录：planner、search、rerank、read、compress 的每一步都会被记录。
- RAG 评测：提供确定性的 retrieval hit rate、evidence recall、groundedness、skill F1、token usage 和 tool-call 指标。
- 动态知识库维护：支持岗位验证、stale 检测、soft archive、audit log、去重和索引刷新。
- Mock 优先的外部接口：网页搜索和页面解析支持 `mock://` 证据，离线也能跑通。

## 系统架构

```mermaid
flowchart LR
  Q[用户问题] --> A[ReAct Agent]
  A --> P[Planner]
  P --> T[Tool Registry]
  T --> K[keyword_search]
  T --> S[semantic_search]
  T --> H[hybrid_search]
  H --> R[rerank_context]
  R --> C[chunk_read]
  C --> B[Context Budget Manager]
  B --> X[context_compression]
  X --> G[Answer Generator]
  G --> O[Citations + Retrieval Trace]
  O --> E[Evaluation]
```

动态维护流程：

```mermaid
flowchart LR
  Q[用户查询] --> R[检索候选岗位]
  R --> F[检查 freshness 元数据]
  F --> V[验证原始 URL 或网页证据]
  V --> P[解析页面状态信号]
  P --> U[更新岗位元数据]
  U --> A[Soft archive 已关闭岗位]
  U --> I[刷新受影响索引]
  I --> O[返回已验证的 active 岗位和引用]
```

## 数据来源

- `data/raw/jobs.csv`：mock 岗位知识库，覆盖 active、closed、stale、unknown 和 duplicate 状态。
- `data/raw/resume.md`：包含 `akashic-agent` 经验的候选人简历。
- `data/raw/company_docs/`：公司资料 markdown。
- `data/raw/interview_notes/`：面试准备笔记。
- `data/eval/qa_eval.json`：RAG 问答评测集。
- `data/eval/job_verification_eval.json`：岗位验证评测集。

## 快速开始

如需可编辑安装：

```bash
python -m pip install -e .
```

构建索引：

```bash
python -m kairorag.indexing.build_index --input data/raw --output data/indexes
```

运行单条查询：

```bash
python -m kairorag.demos.run_single_query \
  --question "哪些岗位要求 RAG、LangGraph 或多 Agent 编排经验？"
```

查询并验证岗位状态：

```bash
python -m kairorag.demos.run_single_query \
  --question "只返回目前仍在招聘、并且要求 RAG 或多 Agent 编排经验的岗位" \
  --verify-jobs
```

运行 RAG 评测：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/qa_eval.json \
  --output results/eval_report.json
```

运行岗位验证评测：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/job_verification_eval.json \
  --output results/job_verification_eval_report.json \
  --job-verification --mock-web
```

批量验证岗位：

```bash
python -m kairorag.maintenance.job_verifier \
  --jobs-file data/raw/jobs.csv \
  --status unknown,stale \
  --limit 5
```

## 示例问题

1. 哪些岗位要求 RAG 或知识库问答经验？
2. 哪些岗位更看重 LangGraph / 多 Agent 编排？
3. 我的 akashic-agent 项目能匹配哪些 JD 要求？
4. 我还缺哪些技能点？
5. 只返回目前仍在招聘、并且要求 RAG 或多 Agent 编排经验的岗位。

## 评测指标

KairoRAG 使用确定性指标，因此无需 LLM judge 也能运行：

- `retrieval_hit_rate`：gold chunk 是否至少有一个出现在 search results 或 read chunks 中。
- `evidence_recall`：`chunk_read` 覆盖了多少 gold chunks。
- `groundedness_score`：citations 是否来自 read chunks，且是否覆盖 gold 证据。
- `skill_extraction_f1`：预测技能与期望技能的匹配程度。
- `context_token_usage`：读取证据后的上下文 token 估计值。
- `tool_call_count`：检索与验证动作的调用次数。
- `verification_accuracy`：岗位状态判断准确率。
- `active_precision`：active 岗位预测精度。
- `closed_recall`：closed 岗位召回率。
- `evidence_domain_match`：证据来源是否符合预期的 mock/web 域。

## 动态知识库维护

企业知识库会过期：岗位会关闭、换链接、跨平台重复发布，或者变成无法确认的 stale 状态。KairoRAG 通过基于证据的维护流程处理这些问题：

- `active`：当前页面或搜索证据显示仍可申请。
- `closed`：可靠页面证据显示岗位已关闭或已招满。
- `stale`：证据不足，因此系统明确标记为“不确定”。
- `updated`：旧链接失效，但搜索发现了新的 active 岗位页。
- `duplicate`：该记录指向主记录，默认检索时不返回。

Closed 和 duplicate 岗位不会被物理删除，只会 soft archive 或标记为 inactive，然后从默认检索中排除。更新流程支持 dry-run，也支持通过 `--apply-updates` 实际写回；只有实际写回时才会把记录追加到 `data/maintenance/audit_log.jsonl`。

当前索引刷新器采用 fallback rebuild 策略，先保证行为正确，再把真正的增量刷新留给后续优化。

## 简历亮点

英文版：

KairoRAG: Agentic Knowledge Retrieval System. Built an Agentic RAG system for JD, resume, company, and interview knowledge bases. Designed hierarchical retrieval tools including keyword_search, semantic_search, chunk_read, rerank_context, and context_compression, enabling a ReAct-style agent to plan multi-step retrieval instead of relying on fixed top-k vector search. Implemented chunk-level citations, retrieval traces, context budget control, and RAG evaluation metrics such as retrieval hit rate, groundedness, skill extraction F1, tool call count, and context token usage. Extended the system with dynamic knowledge-base maintenance, allowing the agent to verify job freshness from web evidence, mark stale/closed records, soft-archive invalid entries, and refresh affected indexes.

中文版：

KairoRAG：企业知识库 Agentic RAG 检索、评测与动态维护系统。基于 Python 构建面向 JD、简历、公司资料和面试材料的 Agentic RAG 系统，将检索过程拆分为 keyword_search、semantic_search、chunk_read、rerank_context 和 context_compression 等层级工具，由 ReAct-style Agent 自主规划多轮检索路径，避免固定 top-k 向量检索带来的无关上下文注入；实现 chunk-level citation、retrieval trace、context budget 控制和 RAG eval，评估 retrieval hit rate、groundedness、技能抽取 F1、tool call count 与 context token usage；扩展动态知识库维护流程，使 Agent 能基于网页证据验证岗位是否仍在招聘，自动标记 active/closed/stale/updated/duplicate 状态，对已关闭岗位进行 soft archive 并刷新受影响索引。

## 面试讲解要点

### 1. 为什么不用普通 top-k RAG？

普通 top-k RAG 对公司名、岗位名、技术关键词等专有名词不稳定，而且容易把无关 chunk 塞进 prompt。KairoRAG 把检索拆成 `keyword_search`、`semantic_search`、`hybrid_search` 和 `chunk_read`，先多轮检索，再读取证据原文。

### 2. Agentic RAG 和普通 RAG 的区别是什么？

普通 RAG 是固定 pipeline；Agentic RAG 让 Agent 根据当前问题和中间检索结果，自主决定下一步调用哪个检索工具、是否继续读取 chunk、是否需要补充检索或验证岗位状态。

### 3. 如何处理长上下文？

通过 `ContextBudgetManager` 限制 search results、read chunks、tool calls 和 context tokens，并进行去重、rerank、compression，不把所有检索结果直接塞进 prompt。

### 4. 如何降低幻觉？

最终回答必须基于 `chunk_read` 读到的证据，或者保留 citation 的压缩上下文；必须输出 chunk-level citations。如果没有足够证据，就明确回答“现有知识库中没有足够证据”。

### 5. 如何评测？

通过 `qa_eval.json` 评估 `retrieval_hit_rate`、`evidence_recall`、`groundedness_score`、`skill_extraction_f1`、`context_token_usage`、`tool_call_count` 和 `latency_ms`。岗位验证则使用 `verification_accuracy`、`active_precision`、`closed_recall` 等指标。

### 6. 动态知识库维护解决什么问题？

企业知识库会过期，岗位可能已经关闭。KairoRAG 会检查 `last_verified_at` 和 `verification_status`，必要时访问原始链接或搜索网页证据验证岗位状态；closed 岗位会被 soft archive，并从 active retrieval 中剔除，避免 RAG 返回过期岗位。

### 7. 和 akashic-agent 怎么区分？

`akashic-agent` 证明的是 Agent 框架、MCP、插件系统、主动任务和长期记忆能力；KairoRAG 证明的是企业级 RAG 能力，包括索引、检索策略、证据引用、上下文预算、RAG eval 和动态知识库维护。

