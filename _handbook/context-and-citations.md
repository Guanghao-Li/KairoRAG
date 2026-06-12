# Context And Citations

## 先理解它是什么

KairoRAG 用 `ContextBudgetManager` 控制检索和读取规模，用 `compress_context()` 做抽取式压缩，用 `answer_generator` 生成带 chunk-level citations 的回答。

它不是把所有 top-k 结果塞进 prompt，而是在有限上下文内保留可追溯证据。

## 它解决什么问题

RAG 系统常见的问题有三个：

1. 检索结果过多，上下文膨胀。
2. 只保留摘要，丢掉原始证据 ID。
3. 回答里的引用无法追溯到真正读过的 chunk。

KairoRAG 的约束是：回答中的引用必须能追溯到 read chunks；如果压缩了上下文，也必须保留 `chunk_id`。

## 系统怎么跑

一条查询从搜索到回答的流程：

```text
question
  -> search results: chunk_id + snippet
  -> limit_search_results: 去重、截断
  -> rerank_context: 选择更相关候选
  -> chunk_read: 读取 chunk 原文
  -> record_read: 检查 chunk 数和 token 预算
  -> compress_context: 抽取相关句子，保留 chunk_id
  -> generate_answer: 输出 answer + citations
```

预算控制项来自 `kairorag/config.py`：

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `max_search_results` | 10 | 限制进入后续阶段的搜索结果数 |
| `max_chunks_to_read` | 5 | 限制最多读取多少个 chunk |
| `max_context_tokens` | 1800 | 限制已读证据的估算 token 数 |
| `max_tool_calls` | 20 | 限制 agent 工具调用次数 |

## 关键设计点

压缩是抽取式的：`compress_context()` 会把 query terms 和 chunk 句子做匹配，每个 chunk 最多保留若干句。输出块包含：

```json
{
  "chunk_id": "job_001_chunk_000",
  "title": "Northstar AI - AI Agent RAG Engineer",
  "source_type": "job",
  "text": "..."
}
```

`generate_answer()` 会为每个有效 read chunk 生成 citation：

```json
{
  "chunk_id": "job_001_chunk_000",
  "source_type": "job",
  "title": "...",
  "evidence": "..."
}
```

eval 里的 `groundedness_score` 会检查 citation IDs 是否都来自 read chunks。如果引用了没有读过的 chunk，分数会变成 `0.0`。

## 怎么运行 / 怎么验证

控制读取和工具预算：

```bash
python -m kairorag.demos.run_single_query \
  --question "哪些岗位更看重 RAG evaluation 和 groundedness？" \
  --max-tool-calls 12 \
  --max-chunks-to-read 3 \
  --json
```

运行上下文预算测试：

```bash
python -m pytest tests/test_context_budget.py
```

运行 groundedness 相关指标测试：

```bash
python -m pytest tests/test_eval_metrics.py
```

## 边界与当前实现

- token 统计是本地估算，不是某个模型 tokenizer 的精确计数。
- compression 当前是简单抽取式压缩，不是生成式摘要。
- citation 粒度是 chunk 级，不是字符级或句子级。
- 当前答案生成是确定性规则实现，不调用外部 LLM。
