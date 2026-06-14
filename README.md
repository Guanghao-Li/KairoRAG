# KairoRAG

KairoRAG 是一个 cloud-native Agentic RAG 项目。当前主路径统一通过 `kairo` CLI 运行，覆盖 OpenAI LLM、OpenAI embedding、Qdrant、BM25、hybrid RRF、cloud reranker、真实 Web Search、岗位 freshness verification、Qdrant metadata 安全写回、observability 和离线 eval dashboard。

旧 offline demo 与旧 pickle index 不再是支持入口。旧 demo CLI、pickle index 和 legacy ReactAgent 只保留为历史参考，不再作为推荐入口。

## 快速开始

```bash
python -m pip install -e ".[dev,web]"
cp .env.example .env
docker compose --profile dev up -d qdrant
kairo doctor --repo
kairo index
kairo query "哪些岗位要求 RAG？"
kairo eval --suite all --fake-providers
```

没有安装 entry point 时也可以使用：

```bash
python -m kairorag.cli --help
python -m kairorag.cli eval --suite all --fake-providers
```

## Production Readiness

- 仓库卫生检查：`kairo doctor --repo`
- 发布前检查：`kairo doctor --release`
- 发布清单：[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)
- 离线 eval：`kairo eval --suite all --fake-providers`
- eval 对比：`kairo eval --compare baseline.json current.json --output-dir data/eval_compare`
- 本地 UI：`kairo ui --host 127.0.0.1 --port 8000`
- 部署指南：[deploy/README.md](deploy/README.md)

`doctor --repo` 不调用外部服务，会检查 `.env`、疑似 secret、未提交或未跟踪文件、生成产物、`.gitignore` 规则、legacy 主路径 import、`kairo` entry point、CI、Docker、Compose、`.env.example` 和 Makefile。

## Approval / Writeback Safety

metadata 写回默认受审批层保护：

- `APPROVAL_POLICY=require_confirmation` 是默认策略。
- `QDRANT_METADATA_WRITE_DRY_RUN=true` 默认只演练。
- CLI 真实写回必须显式传 `--yes`，并且 Qdrant 写回开关必须启用。
- Agent 工具即使请求 `dry_run=false`，也不能自主真实写回。
- audit log 会记录 `approval_decision`、`dry_run`、`applied` 和 metadata patch 摘要。

支持策略：

- `deny`：拒绝所有破坏性操作。
- `dry_run`：只允许 dry-run。
- `require_confirmation`：未确认时只 dry-run，确认后才允许真实写回。
- `allow`：仍要求 CLI `--yes` 后才真实写回。

## Live Provider Tests

live tests 默认全部跳过，不在普通 `python -m pytest` 或默认 CI 中真实调用外部服务。

```bash
RUN_LIVE_PROVIDER_TESTS=1 LIVE_PROVIDER=openai python -m pytest tests/live -m live
RUN_LIVE_PROVIDER_TESTS=1 LIVE_PROVIDER=qdrant python -m pytest tests/live -m live
```

GitHub Actions 中的 `Live Provider Contract Tests` 只能通过 `workflow_dispatch` 手动触发，并通过 GitHub Secrets 注入 provider keys。

## Eval Comparison

先生成两份离线结果：

```bash
kairo eval --suite all --fake-providers --tag baseline --output-dir results/baseline
kairo eval --suite all --fake-providers --tag current --output-dir results/current
```

再比较：

```bash
kairo eval --compare results/baseline/eval_results.json results/current/eval_results.json --output-dir data/eval_compare
```

输出包含 `eval_comparison.md`、`eval_comparison.html` 和可追加历史记录的 `trend.json`。

## UI Dashboard

```bash
kairo ui --host 127.0.0.1 --port 8000
```

UI 会展示项目状态、eval metrics、trace、audit 和 doctor 报告。它没有身份系统，默认只建议本地使用。不要把它直接暴露到公网。

## Deployment

Docker Compose profiles：

- `dev`：本地 Qdrant。
- `app`：app 和 UI smoke 服务。
- `eval`：离线 eval 容器。

常用命令：

```bash
docker compose --profile dev up -d qdrant
docker compose --profile app up --build ui
docker compose --profile eval run --rm eval
```

生产部署请参考 [deploy/README.md](deploy/README.md)，其中包含 systemd、Cloud Run、Render 和 Railway 的说明。真实 secret 只放在部署平台的密钥系统中，不提交 `.env`。

## CI

默认 CI 包含：

- `ruff check .`
- secret scan
- import smoke
- CLI smoke
- `coverage run -m pytest`
- fake eval suite smoke
- 上传 eval dashboard 和 coverage artifact

pre-commit 包含 ruff、trailing whitespace、end of file fixer、check yaml、detect private key 和 check added large files。

## Makefile

```bash
make install
make test
make lint
make ci
make doctor-release
make eval
make eval-compare
make ui
make live-test
make docker-ui
make docker-eval
```

## 环境变量

请从 `.env.example` 复制 `.env`，并只在本地或部署环境中填入真实 secret。核心分组：

- LLM / embedding：`OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`、`OPENAI_EMBEDDING_MODEL`
- Qdrant：`QDRANT_URL`、`QDRANT_API_KEY`、`QDRANT_COLLECTION`
- Web Search：`TAVILY_API_KEY`、`SERPAPI_API_KEY`、`BING_API_KEY`
- Reranker：`COHERE_API_KEY`、`JINA_API_KEY`、`VOYAGE_API_KEY`
- Approval：`APPROVAL_POLICY`、`REQUIRE_HUMAN_APPROVAL_FOR_WRITEBACK`
- Observability：`TRACE_LOG_PATH`、`FRESHNESS_AUDIT_LOG_PATH`

## 测试

```bash
python -m ruff check .
python -m pytest
python -m kairorag.cli --help
python -m kairorag.cli doctor --repo
python -m kairorag.cli doctor --release
python -m kairorag.cli eval --suite all --fake-providers
python -m kairorag.cli ui --help
```

## Legacy 状态

以下入口不再推荐作为主路径：

- `python -m kairorag.demos.run_single_query`
- `python -m kairorag.demos.run_batch_eval`
- `python -m kairorag.indexing.build_index`
- `python -m kairorag.scripts.build_index`

cloud 主路径不得 fallback 到 legacy ReactAgent、legacy planner、mock web search、deterministic answer generator、pickle VectorStore 或 hashing embedding。
