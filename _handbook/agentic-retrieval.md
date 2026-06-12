# Agentic Retrieval

## 先理解它是什么

KairoRAG 的检索不是固定的 `vector search top-k -> answer`。当前实现使用规则 planner 判断问题类型，再调用本地检索工具、重排候选、读取 chunk 原文，并把这些动作写入 retrieval trace。

核心文件：

- `kairorag/agent/planner.py`
- `kairorag/agent/react_agent.py`
- `kairorag/retrieval/keyword_search.py`
- `kairorag/retrieval/semantic_search.py`
- `kairorag/retrieval/hybrid_search.py`
- `kairorag/retrieval/reranker.py`
- `kairorag/retrieval/chunk_read.py`

## 它解决什么问题

普通 top-k RAG 容易把搜索结果 snippet 当成事实，也容易因为只走一种检索方式漏掉专有名词或语义相关内容。KairoRAG 把过程拆开：

```text
question
  -> planner
  -> keyword_search / semantic_search / hybrid_search
  -> rerank_context
  -> chunk_read
  -> compression with chunk_id
  -> grounded answer with citations
```

这样做的重点不是“工具更多”，而是明确区分线索和证据：search result 的 snippet 只是候选线索，最终回答必须来自 `chunk_read` 读到的 chunk，或来自保留原始 `chunk_id` 的压缩证据块。

## 系统怎么跑

1. `ReactAgent.run()` 先调用 `ensure_indexes()`，本地索引缺失时会基于 `data/raw` 构建索引。
2. `plan_query()` 根据问题生成 `Plan`。
3. agent 按 plan 选择一个搜索工具。
4. `ContextBudgetManager.limit_search_results()` 对结果去重和截断。
5. `rerank_context()` 重新排序候选 chunk。
6. agent 对重排后的候选逐个调用 `chunk_read()`。
7. `compress_context()` 从已读 chunk 中抽取相关句子，并保留 `chunk_id`。
8. `generate_answer()` 只基于已读 chunk 和压缩证据生成答案与 citations。

## 关键设计点

Planner 当前是规则实现，不是外部 LLM planner：

- 如果用户显式传入 `--verify-jobs`，或问题包含“还在招”“正在招聘”“closed”“fresh”等 freshness 词，选择 `hybrid_search` 并设置 `requires_verification=True`。
- 如果问题包含 `rag`、`langgraph`、`mcp`、`akashic-agent`、公司名、岗位等关键词，选择 `hybrid_search`。
- 其他偏概念型问题默认走 `semantic_search`。

工具分工：

| 工具 | 当前作用 |
| --- | --- |
| `keyword_search` | 适合公司名、项目名、技能名等精确词 |
| `semantic_search` | 适合概念型、语义相近的问题 |
| `hybrid_search` | 融合关键词和语义结果，当前用于大部分技能/JD/freshness 查询 |
| `rerank_context` | 在候选结果内重新排序，控制进入读取阶段的 chunk |
| `chunk_read` | 读取 chunk 原文，并可带前后窗口 |

## 怎么运行 / 怎么验证

构建索引：

```bash
python -m kairorag.indexing.build_index --input data/raw --output data/indexes
```

运行查询并查看 trace：

```bash
python -m kairorag.demos.run_single_query \
  --question "哪些岗位要求 RAG、LangGraph 或多 Agent 编排经验？" \
  --json
```

运行相关测试：

```bash
python -m pytest tests/test_retrieval.py
```

## 边界与当前实现

- planner 是规则判断，不是 LLM function calling 或 LangGraph workflow。
- 当前 search 阶段会返回 snippet，但 snippet 不应被当成最终事实来源。
- `chunk_read` 才是回答证据入口；压缩后的证据也必须保留 `chunk_id`。
- 当前实现优先保证本地可运行和可测试，不依赖付费 API。
