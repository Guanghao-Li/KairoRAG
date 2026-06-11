# KairoRAG 岗位验证评测报告

## 总体指标

| 指标 | 数值 |
| --- | ---: |
| verification_accuracy | 1.000 |
| active_precision | 1.000 |
| closed_recall | 1.000 |
| stale_rate | 0.200 |
| update_correctness | 1.000 |
| evidence_domain_match | 1.000 |
| reindex_trigger_accuracy | 1.000 |

## 逐条样例

### ver_001
- 检索路径：parse_job_page -> original_url
- 预测结果：`active`
- 指标明细：`{"confidence": 0.94, "needs_reindex": false}`

### ver_002
- 检索路径：parse_job_page -> original_url
- 预测结果：`closed`
- 指标明细：`{"confidence": 0.93, "needs_reindex": false}`

### ver_003
- 检索路径：parse_job_page -> search_web
- 预测结果：`stale`
- 指标明细：`{"confidence": 0.25, "needs_reindex": true}`

### ver_004
- 检索路径：parse_job_page -> search_web
- 预测结果：`updated`
- 指标明细：`{"confidence": 0.9, "needs_reindex": true}`

### ver_005
- 检索路径：parse_job_page -> original_url
- 预测结果：`duplicate`
- 指标明细：`{"confidence": 0.88, "needs_reindex": true}`
