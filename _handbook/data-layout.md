# Data Layout

## 先理解它是什么

KairoRAG 的数据目录分成原始知识库、索引、评测集、运行结果和维护日志。项目默认从 `data/raw` 构建索引，从 `data/eval` 读取 eval 数据，把结果写到 `results`。

## 它解决什么问题

RAG 项目容易把原始文档、索引产物、评测数据和运行报告混在一起。KairoRAG 当前的数据布局让每类文件职责明确：

```text
data/
  raw/          原始知识库
  indexes/      build_index 生成的本地索引
  eval/         确定性 eval 数据集
  maintenance/  apply 模式下的 audit log
results/        eval 输出报告
```

## 系统怎么跑

构建索引时：

1. `load_knowledge_base(data/raw)` 读取岗位、简历、公司文档和面试笔记。
2. `chunk_documents()` 生成 chunks。
3. `KeywordIndex` 保存为 `keyword_index.pkl`。
4. `VectorStore` 保存为 `vector_store.pkl`。
5. 同步写出 `chunks.json`、`documents.json`、`jobs.json` 方便检查。

## 关键设计点

原始数据：

| 路径 | 用途 |
| --- | --- |
| `data/raw/jobs.csv` | 岗位知识库，包含技能、URL、状态、证据、重复关系和优先级 |
| `data/raw/resume.md` | 候选人简历，用于回答项目/JD 匹配问题 |
| `data/raw/company_docs/` | 公司资料 markdown |
| `data/raw/interview_notes/` | 面试笔记 markdown |

索引产物：

| 路径 | 用途 |
| --- | --- |
| `data/indexes/keyword_index.pkl` | 关键词检索索引 |
| `data/indexes/vector_store.pkl` | 本地向量检索存储 |
| `data/indexes/chunks.json` | chunk 明细 |
| `data/indexes/documents.json` | 文档明细 |
| `data/indexes/jobs.json` | 岗位记录快照 |

评测数据：

| 路径 | 用途 |
| --- | --- |
| `data/eval/qa_eval.json` | QA eval 样例，包含问题、期望技能和 gold chunk IDs |
| `data/eval/job_verification_eval.json` | 岗位验证 eval 样例，包含期望状态和是否应触发 reindex |

## 怎么运行 / 怎么验证

重建索引：

```bash
python -m kairorag.indexing.build_index --input data/raw --output data/indexes
```

查看索引统计输出，正常会打印文档数、岗位数和 chunk 数。

运行 eval 并生成报告：

```bash
python -m kairorag.demos.run_batch_eval \
  --eval-file data/eval/qa_eval.json \
  --output results/eval_report.json
```

## 如何扩展自己的知识库

扩展岗位数据：

1. 在 `data/raw/jobs.csv` 添加行。
2. 保持 `job_id` 唯一。
3. 用 `|` 分隔 `required_skills`、`preferred_skills` 和 `agent_related_keywords`。
4. 如果暂时无法验证状态，把 `verification_status` 设为 `unknown`。
5. 运行 `build_index` 重建索引。

扩展文档数据：

1. 在 `data/raw/company_docs/` 或 `data/raw/interview_notes/` 添加 markdown。
2. 文件名会参与生成文档 ID。
3. 运行 `build_index`。
4. 如需评测，向 `data/eval/qa_eval.json` 添加问题和对应 gold chunk IDs。

## 边界与当前实现

- 当前索引是本地文件产物，不依赖外部向量数据库。
- `vector_store.pkl` 使用项目内本地 embedding/tokenize 逻辑，不调用付费 embedding API。
- 修改原始数据后需要手动运行 `build_index`，或通过维护流程触发 fallback rebuild。
- `data/maintenance/audit_log.jsonl` 只有在 apply 更新时才会出现。
