# Evaluation

## 先理解它是什么

KairoRAG 的 eval 是确定性的，不依赖 LLM judge。它用固定数据集、agent 运行结果和显式指标计算 RAG 检索质量、证据覆盖、引用可追溯性、技能抽取效果、上下文使用量和工具调用量。

相关文件：

- `data/eval/qa_eval.json`
- `data/eval/job_verification_eval.json`
- `kairorag/evaluation/metrics.py`
- `kairorag/evaluation/evaluator.py`
- `kairorag/demos/run_batch_eval.py`

## 它解决什么问题

RAG 的主观答案质量很难稳定评估。KairoRAG 当前先评估能确定计算的工程指标：

```text
eval item
  -> run ReactAgent
  -> collect retrieval_trace / read_chunks / citations / metrics
  -> compute deterministic metrics
  -> write JSON report
  -> write Markdown report
```

这样可以快速发现检索漏召、没有读到 gold evidence、引用不 grounded、上下文过大或工具调用过多等问题。

## 系统怎么跑

QA eval 使用 `data/eval/qa_eval.json`。每条样例包含：

- `id`
- `question`
- `expected_skills`
- `gold_chunk_ids`

岗位验证 eval 使用 `data/eval/job_verification_eval.json`。每条样例包含：

- `job_id`
- `expected_status`
- `expected_needs_reindex`
- `expected_evidence_domain`

`run_batch_eval` 会输出 JSON 报告，并在同目录生成同名 Markdown 报告。

## 关键设计点

QA 指标：

| 指标 | 用来发现什么问题 |
| --- | --- |
| `retrieval_hit_rate` | gold chunk 是否至少被搜索到或读到 |
| `evidence_recall` | `chunk_read` 是否覆盖了 gold chunks |
| `groundedness_score` | citations 是否来自 read chunks，并且是否命中 gold evidence |
| `skill_extraction_precision` | 抽取出的技能是否混入太多无关项 |
| `skill_extraction_recall` | 是否漏掉期望技能 |
| `skill_extraction_f1` | 技能抽取的综合效果 |
| `context_token_usage` | 读入证据后的上下文规模 |
| `tool_call_count` | agent 检索与维护动作是否过多 |
| `latency_ms` | 单条样例运行耗时 |

岗位验证指标：

| 指标 | 用来发现什么问题 |
| --- | --- |
| `verification_accuracy` | 岗位状态整体判断是否正确 |
| `active_precision` | 被判为 active 的岗位是否可靠 |
| `closed_recall` | closed 岗位是否被找出来 |
| `stale_rate` | 有多少岗位无法确认 |
| `update_correctness` | updated 状态和字段更新是否正确 |
| `evidence_domain_match` | evidence 是否来自预期 mock/web 域 |
| `reindex_trigger_accuracy` | 需要刷新索引的判断是否正确 |

## 怎么运行 / 怎么验证

运行 QA eval：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/qa_eval.json \
  --output results/eval_report.json
```

带 trace 输出：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/qa_eval.json \
  --output results/eval_report.json \
  --include-traces
```

运行岗位验证 eval：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/job_verification_eval.json \
  --output results/job_verification_eval_report.json \
  --job-verification
```

运行指标测试：

```bash
python -m pytest tests/test_eval_metrics.py
```

## 边界与当前实现

- 当前 eval 不判断自然语言答案是否“写得好”，只判断可计算的检索、证据、引用和维护指标。
- `skill_extraction_f1` 基于当前答案生成里的固定技能词表，不是通用 NER。
- `--mock-web` 参数存在于 `run_batch_eval` CLI，但当前 evaluator 没有读取该参数；岗位验证默认使用项目内 mock web evidence。
- 报告会写到 `results/` 下，路径由 `--output` 控制。
