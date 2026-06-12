# Dynamic Maintenance

## 先理解它是什么

KairoRAG 的动态维护用于处理岗位知识库 stale 的问题。它会基于原始 URL、mock web search 和页面状态解析，判断岗位是否还有效，并在需要时 dry-run 或写回更新。

核心文件：

- `kairorag/maintenance/job_verifier.py`
- `kairorag/maintenance/job_page_parser.py`
- `kairorag/maintenance/web_search.py`
- `kairorag/maintenance/knowledge_base_updater.py`
- `kairorag/maintenance/deduplicator.py`
- `kairorag/maintenance/index_refresher.py`
- `kairorag/maintenance/audit_log.py`

## 它解决什么问题

企业知识库会 stale，岗位数据尤其明显：

- 岗位已经关闭，但检索仍返回。
- 原始链接失效，但新链接还在招。
- 第三方平台重复发布同一个岗位。
- 页面证据不足，系统无法确认状态。

当前状态枚举来自 `VerificationStatus`：

| 状态 | 含义 |
| --- | --- |
| `active` | 原始页面或搜索证据显示岗位仍可申请 |
| `closed` | 页面证据显示岗位关闭、招满或不可用 |
| `stale` | 无法确认 active 或 closed |
| `unknown` | 原始数据中尚未确认的状态 |
| `updated` | 旧 URL 失效或不确定，但搜索发现新的 active URL |
| `duplicate` | 记录被标记为另一个岗位的重复项 |

## 系统怎么跑

岗位验证主流程：

```text
job record
  -> if duplicate: return duplicate result
  -> parse original job_url
  -> if active/closed: return status with evidence
  -> search_web(query)
  -> parse returned mock pages
  -> if new active URL: return updated + updated_fields
  -> otherwise: return stale
```

当 agent 查询触发 freshness 需求时：

1. 先执行正常检索和 `chunk_read`。
2. 从已读 job chunks 中收集 candidate job IDs。
3. `needs_verification()` 判断是否需要验证。
4. 调用 `verify_job()` 产生 `JobVerificationResult`。
5. 调用 `apply_verification_result()`，默认 dry-run。
6. 如果用户传入 `--apply-kb-updates`，才写回 CSV。
7. 如果写回且需要重建索引，调用 `refresh_index_for_jobs()`。

## 关键设计点

Dry-run 和 apply 的区别：

| 模式 | 命令参数 | 行为 |
| --- | --- | --- |
| dry-run | 默认 | 返回会变更的字段，但不改 `jobs.csv`，不追加审计日志 |
| apply | `--apply-updates` 或 `--apply-kb-updates` | 写回 `jobs.csv`，追加 `data/maintenance/audit_log.jsonl` |

Soft archive 的语义：

- `closed` 岗位不会被物理删除。
- 写回时会设置 `closed_reason`。
- soft 模式下会设置 `archived_at`。
- 默认检索会排除 `closed` 和 `duplicate`，但可以通过检索工具参数 `include_archived=True` 查到。

索引刷新：

- 当前 `refresh_index_for_jobs()` 是 fallback rebuild。
- 它会重新调用 `build_indexes(RAW_DATA_DIR, INDEX_DIR)`。
- 当前没有实现真正的增量索引更新。

## 怎么运行 / 怎么验证

验证单个岗位：

```bash
python -m kairorag.maintenance.job_verifier \
  --jobs-file data/raw/jobs.csv \
  --job-id job_008
```

批量验证 unknown / stale 岗位，默认 dry-run：

```bash
python -m kairorag.maintenance.job_verifier \
  --jobs-file data/raw/jobs.csv \
  --status unknown,stale \
  --limit 5
```

写回更新：

```bash
python -m kairorag.maintenance.job_verifier \
  --jobs-file data/raw/jobs.csv \
  --job-id job_008 \
  --apply-updates
```

通过 agent 查询触发验证，但不写回：

```bash
python -m kairorag.demos.run_single_query \
  --question "只返回目前仍在招聘、并且要求 RAG 或多 Agent 编排经验的岗位" \
  --verify-jobs
```

通过 agent 查询触发验证并写回：

```bash
python -m kairorag.demos.run_single_query \
  --question "只返回目前仍在招聘、并且要求 RAG 或多 Agent 编排经验的岗位" \
  --verify-jobs \
  --apply-kb-updates
```

运行维护测试：

```bash
python -m pytest tests/test_job_verifier.py tests/test_kb_updater.py
```

## 边界与当前实现

- `search_web()` 当前是 mockable、本地返回的 evidence，不接真实搜索 API。
- 页面解析主要基于 mock URL 和页面状态信号。
- `job_verifier` 中 `--search-web` 参数存在，但当前实现实际总是搜索，因为调用使用了 `search=args.search_web or True`。
- fallback index rebuild 是当前策略，不是高阶增量索引。
- `duplicate` 既可以来自原始 CSV，也可以由 `deduplicator` 检测并在 apply 模式下标记。
