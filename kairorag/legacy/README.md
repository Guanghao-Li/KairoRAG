# Legacy 离线实现

本目录用于说明阶段一重构后仍被保留的历史实现。旧代码没有删除，是为了让后续迁移可以对照行为和测试结果。

以下模块已经被标记为 Legacy / Deprecated：

- `kairorag.indexing.embedding.HashingEmbeddingProvider`：本地 hashing embedding。
- `kairorag.indexing.vector_store.VectorStore`：pickle-backed 本地向量库。
- `kairorag.indexing.keyword_index.KeywordIndex`：keyword overlap 本地关键词索引。
- `kairorag.agent.planner.plan_query`：rule-based planner。
- `kairorag.maintenance.web_search.search_web`：mock web search。
- `kairorag.agent.answer_generator.generate_answer`：确定性答案生成器。

新的 cloud 主路径不会默认 fallback 到这些模块。需要真实运行 cloud runtime 时，必须通过 `KairoCloudSettings` 提供 OpenAI、Qdrant 和 web search provider 的必要环境变量。
