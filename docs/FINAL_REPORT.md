# KairoRAG 最终交付报告

## 项目完成情况

KairoRAG 已经实现为一个可本地运行的 Python 3.10+ 后端项目，包含 ingestion、chunking、indexing、retrieval tools、ReAct-style Agentic RAG loop、动态岗位验证、知识库更新、确定性评测、测试、示例文件，以及适合简历和面试讲解的 README 材料。

## 从 A-RAG 借鉴的部分

- 分层检索接口：关键词检索、语义检索和 chunk 阅读。
- 迭代式 Agent 循环：先检索、再观察、再阅读、再判断是否继续。
- 检索轨迹和已读 chunk 跟踪作为一等输出。
- 把上下文效率和证据约束作为核心设计目标。

## KairoRAG 新增的能力

- 面向 JD、简历、公司资料、面试材料的统一知识库 schema。
- 不依赖付费 API 的本地 hashing embedding 和 pickle 向量存储。
- Hybrid search 和确定性 reranking。
- 通过 `chunk_read` 强制执行 chunk-level citations。
- 对搜索结果数、已读 chunks、context tokens、tool calls 的统一预算控制。
- 基于 URL / 页面 / snippet 证据的岗位 freshness 验证。
- 不删除源记录的 soft archive 和 duplicate 处理。
- 对应用型知识库更新的 audit log。
- RAG eval 与岗位验证 eval 双评测体系。

## 核心模块说明

- `kairorag/ingestion`：负责加载文档和 chunk 切分。
- `kairorag/indexing`：负责关键词索引、本地向量索引和 build-index CLI。
- `kairorag/retrieval`：负责关键词检索、语义检索、混合检索、chunk_read、rerank、compression 和 context budget。
- `kairorag/agent`：负责 planner、tool registry、ReAct-style loop 和 answer generator。
- `kairorag/maintenance`：负责 mock web search、页面解析、job verifier、updater、deduplicator、index refresher 和 audit log。
- `kairorag/evaluation`：负责 datasets、metrics、evaluator 和 markdown/json 报告。

## 如何运行

```bash
python -m kairorag.indexing.build_index --input data/raw --output data/indexes
python -m kairorag.demos.run_single_query --question "哪些岗位要求 RAG、LangGraph 或多 Agent 编排经验？"
python -m kairorag.demos.run_single_query --question "只返回目前仍在招聘、并且要求 RAG 或多 Agent 编排经验的岗位" --verify-jobs
python -m kairorag.demos.run_batch_eval --eval-file data/eval/qa_eval.json --output results/eval_report.json
python -m kairorag.maintenance.job_verifier --jobs-file data/raw/jobs.csv --status unknown,stale --limit 5
pytest -q
```

## 测试结果

最近一次本地测试结果：

```text
8 passed
```

核心 CLI 也已经成功运行，并在 `data/indexes/` 和 `results/` 中生成了索引和评测报告。

## 当前限制

- Mock hashing embedding 可复现、便于离线运行，但不等于生产级检索质量。
- Mock web search 仅用于本地测试和演示流程。
- 当前没有真实 LLM judge，评测全部走确定性逻辑。
- 局部索引刷新目前会 fallback 到整库重建。
- 页面解析对复杂的 JavaScript-heavy careers 页面支持有限。

## 后续优化方向

- 增加可选的 OpenAI embedding 或 sentence-transformer embedding provider。
- 实现真正的增量索引刷新，而不只是 fallback rebuild。
- 增强多跳 JD / resume matching 的 query decomposition。
- 在保留 mock/local fallback 的前提下，加入可选 LLM planner 和 LLM judge。
- 增加 Tavily、Exa、Bing、SERP API 等真实搜索 provider 适配层。

