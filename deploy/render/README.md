# Render 部署占位说明

Render 可用于部署 KairoRAG UI 或轻量 app smoke 服务。生产向量库建议使用托管 Qdrant，并通过 Render Environment Secret 注入密钥。

默认保持 `QDRANT_METADATA_WRITE_DRY_RUN=true`，真实写回必须通过审批和 `--yes`。
