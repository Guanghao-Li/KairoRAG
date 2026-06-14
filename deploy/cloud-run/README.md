# Cloud Run 部署说明

Cloud Run 适合运行无状态 API 或 UI 容器。将 Qdrant 使用托管服务或单独的持久化服务，不要把向量库状态写入 Cloud Run 本地磁盘。

建议：

- 使用 Secret Manager 注入 `OPENAI_API_KEY`、`QDRANT_API_KEY` 和 live provider keys。
- 设置 `APPROVAL_POLICY=require_confirmation` 或 `dry_run`。
- UI 容器不要公开给未认证用户；如需访问，请放在 IAP 或受控入口之后。
- 将 trace 和 audit 输出转发到集中日志或挂载的持久存储。
