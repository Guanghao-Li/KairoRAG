# KairoRAG 生产部署指南

KairoRAG 的生产部署应保持可复现、可审计和默认 dry-run。不要把 `.env`、trace、audit、eval output 或临时索引提交到仓库。

## 本地 Docker Compose

- 开发依赖：`docker compose --profile dev up -d qdrant`
- 应用 smoke：`docker compose --profile app up --build app`
- 本地 UI：`docker compose --profile app up --build ui`
- 离线评测：`docker compose --profile eval run --rm eval`

## 生产环境变量

- Provider：`LLM_PROVIDER`、`EMBEDDING_PROVIDER`、`VECTOR_STORE_PROVIDER`、`WEB_SEARCH_PROVIDER`、`RERANKER_PROVIDER`
- OpenAI：`OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`、`OPENAI_EMBEDDING_MODEL`
- Qdrant：`QDRANT_URL`、`QDRANT_API_KEY`、`QDRANT_COLLECTION`
- Web Search：`TAVILY_API_KEY`、`SERPAPI_API_KEY`、`BING_API_KEY`
- Reranker：`COHERE_API_KEY`、`JINA_API_KEY`、`VOYAGE_API_KEY`
- 审批：`APPROVAL_POLICY`、`REQUIRE_HUMAN_APPROVAL_FOR_WRITEBACK`
- 审计和观测：`FRESHNESS_AUDIT_LOG_PATH`、`TRACE_LOG_PATH`

## Secrets 管理

真实密钥只放在部署平台的 Secrets、Kubernetes Secret、systemd EnvironmentFile 或云服务密钥管理器中。仓库只保留 `.env.example`。

## Qdrant 托管与自托管

托管 Qdrant 更适合生产环境，便于备份、监控和访问控制。自托管 Qdrant 需要持久化 `/qdrant/storage`，并对 API key、网络入口和备份策略做独立管理。

## 日志和审计持久化

`FRESHNESS_AUDIT_LOG_PATH` 和 `TRACE_LOG_PATH` 必须挂载到持久卷或集中日志系统。审计日志会记录 approval decision、dry-run、applied 和 metadata patch 摘要。

## UI 安全提醒

`kairo ui` 默认监听 `127.0.0.1`。UI 没有身份系统，不要直接暴露公网。如需远程访问，请放在 VPN、内网、SSH tunnel 或带认证的反向代理之后。
