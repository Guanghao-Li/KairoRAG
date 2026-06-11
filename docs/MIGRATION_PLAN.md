# KairoRAG 迁移计划

## A-RAG 源项目总结

A-RAG 是一个结构紧凑的 `src/arag` Python 包，重点展示 agentic retrieval。它的核心结构包括：

- `src/arag/tools`：分层检索工具，包含 `keyword_search`、`semantic_search` 和 `read_chunk`。
- `src/arag/agent/base.py`：ReAct 风格循环，由 LLM 决定工具调用、观察结果并继续推进，直到能够回答。
- `src/arag/core/context.py`：负责记录检索日志、已读 chunk 和已检索 token 数。
- `scripts/build_index.py`：句子级 embedding 索引构建脚本。
- `scripts/batch_runner.py`：带 checkpoint 输出的并发批处理脚本。
- `scripts/eval.py`：偏基准问答场景的答案评测脚本。
- `src/arag/agent/prompts/default.txt`：要求“检索 -> 阅读 -> 评估 -> 回答”的系统提示。

原项目面向 benchmark RAG 数据集，默认假设存在可选的重依赖，例如 sentence-transformers 和兼容 OpenAI 的 LLM 接口。

## 可复用的思想

- 分层检索接口：对精确术语使用 `keyword_search`，对概念型问题使用 `semantic_search`，对最终证据使用 `chunk_read`。
- 搜索 snippet 不能直接作为最终证据，只有完整读过的 chunk 才能支撑回答。
- ReAct 轨迹日志应当让每一次检索决策都可追踪、可解释。
- 上下文状态要能避免重复读取同一 chunk，并限制 token 膨胀。
- 批量评测不仅要保存最终答案，也要保留轨迹和中间信息。

## 需要重构的模块

- Embedding 必须默认走本地/mock hashing，这样 KairoRAG 在没有 API key 和模型下载的情况下也能运行。
- 数据 schema 需要从 benchmark chunks 改为 JD、简历、公司资料和面试笔记。
- 评测模块需要加入确定性的 groundedness、skill extraction 和岗位验证指标。
- 检索层必须默认过滤 `closed` 和 `duplicate` 岗位。
- Agent 行为需要纳入岗位 freshness 验证和知识库维护流程。
- 索引刷新需要支持 soft archive 语义，而不是删除源记录。

## KairoRAG 目标架构

KairoRAG 不是固定 top-k RAG pipeline。普通 RAG 的流程是：

`question -> vector top-k -> stuff chunks into prompt -> answer`

KairoRAG 把检索暴露为工具：

`question -> planner -> keyword/semantic/hybrid search -> rerank -> chunk_read -> context budget -> compression -> grounded answer with citations`

对于岗位 freshness 相关问题，还会扩展为：

`retrieve candidate jobs -> check freshness/status -> verify URL or mock web evidence -> update metadata -> soft archive closed jobs -> refresh affected indexes`

## 分阶段实现计划

1. 搭建项目骨架、共享 schema、mock 数据和基础文档。
2. 实现 loader、chunking、确定性 keyword index、本地 vector store 和 build-index CLI。
3. 实现检索工具、reranking、抽取式压缩和 context budget 控制。
4. 实现一个基于规则的 ReAct-style agent，记录 retrieval 与 verification trace。
5. 实现动态岗位验证、audit log、soft archive 更新、去重和索引刷新 fallback。
6. 实现确定性评测指标和报告生成。
7. 打磨 README、examples、sample trace 和最终交付报告。

## 关键风险

- 本地 hashing embedding 虽然可复现、便于离线运行，但效果弱于生产级 embedding。
- Mock web search 可以验证工程流程，但不能证明真实岗位 freshness。
- 简单 HTML 解析可能覆盖不到复杂的动态 careers 页面。
- 局部索引刷新当前先用 fallback rebuild，真正增量刷新还需要后续优化。
- 基于规则的 planner 离线可靠，但灵活性仍低于真实 LLM planner。

## 本地依赖

- Python 3.10+
- 核心系统仅依赖标准库
- 可选：`requests` 和 `beautifulsoup4`，用于真实网页解析
- 可选：`pytest`，用于测试

## Phase 0 状态

A-RAG 仅作为参考被读取，没有修改 `D:\CodexProject\arag` 中的任何文件。迁移方案保留了 agentic retrieval 的核心思想，但在实现层面重写为更适合 KairoRAG 企业知识库与动态维护场景的工程结构。

