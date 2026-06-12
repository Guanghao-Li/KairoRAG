"""Cloud-native 离线 eval 指标。"""

from __future__ import annotations

import math
from typing import Any


def retrieval_hit_rate(results: list[dict[str, Any]]) -> float:
    """计算至少命中一个期望 chunk 的 case 占比；空输入返回 0.0。"""

    if not results:
        return 0.0
    hits = 0
    for item in results:
        expected = set(_list(item, "expected_chunk_ids"))
        seen = set(_list(item, "retrieved_chunk_ids")) | set(_list(item, "read_chunk_ids"))
        hits += 1 if expected and expected & seen else 0
    return hits / len(results)


def evidence_recall(results: list[dict[str, Any]]) -> float:
    """计算 read chunks 覆盖期望证据的平均召回率；空输入返回 0.0。"""

    if not results:
        return 0.0
    recalls: list[float] = []
    for item in results:
        expected = set(_list(item, "expected_chunk_ids"))
        if not expected:
            recalls.append(0.0)
            continue
        read = set(_list(item, "read_chunk_ids"))
        recalls.append(len(expected & read) / len(expected))
    return sum(recalls) / len(recalls)


def mean_reciprocal_rank(results: list[dict[str, Any]]) -> float:
    """计算期望 chunk 首次出现位置的 MRR；空输入返回 0.0。"""

    if not results:
        return 0.0
    total = 0.0
    for item in results:
        expected = set(_list(item, "expected_chunk_ids"))
        retrieved = _list(item, "retrieved_chunk_ids")
        rank = 0
        for index, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                rank = index
                break
        total += 1 / rank if rank else 0.0
    return total / len(results)


def ndcg_at_k(results: list[dict[str, Any]], k: int = 5) -> float:
    """计算二元相关性的 nDCG@k；空输入或 k<=0 返回 0.0。"""

    if not results or k <= 0:
        return 0.0
    scores: list[float] = []
    for item in results:
        expected = set(_list(item, "expected_chunk_ids"))
        retrieved = _list(item, "retrieved_chunk_ids")[:k]
        dcg = 0.0
        for index, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                dcg += 1 / math.log2(index + 1)
        ideal_hits = min(len(expected), k)
        idcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores)


def citation_validity(answer: dict[str, Any] | str) -> float:
    """检查 citation 是否全部来自已读取 chunk；无 citation 视为有效。"""

    if isinstance(answer, str):
        return 1.0
    citations = set(_list(answer, "citations"))
    read = set(_list(answer, "read_chunk_ids"))
    return 1.0 if citations <= read else 0.0


def invalid_citation_rate(answers: list[dict[str, Any]]) -> float:
    """计算所有 citation 中未被读取 chunk 覆盖的比例；无 citation 返回 0.0。"""

    total = 0
    invalid = 0
    for answer in answers:
        read = set(_list(answer, "read_chunk_ids"))
        for citation in _list(answer, "citations"):
            total += 1
            if citation not in read:
                invalid += 1
    return invalid / total if total else 0.0


def snippet_leakage_rate(answers: list[dict[str, Any]]) -> float:
    """计算把 web/search snippet 当作最终证据的答案比例；空输入返回 0.0。"""

    if not answers:
        return 0.0
    leaks = sum(1 for item in answers if bool(item.get("used_search_snippet_as_evidence") or item.get("snippet_leakage")))
    return leaks / len(answers)


def no_evidence_refusal_rate(answers: list[dict[str, Any]]) -> float:
    """计算无 evidence 场景下安全拒答的比例；没有无证据 case 返回 0.0。"""

    no_evidence = [item for item in answers if not _list(item, "read_chunk_ids")]
    if not no_evidence:
        return 0.0
    refused = sum(1 for item in no_evidence if bool(item.get("refused_without_evidence")))
    return refused / len(no_evidence)


def planning_success_rate(agent_results: list[dict[str, Any]]) -> float:
    """计算 Agent planning 成功率；空输入返回 0.0。"""

    if not agent_results:
        return 0.0
    return sum(1 for item in agent_results if bool(item.get("planning_success"))) / len(agent_results)


def invalid_tool_call_rate(agent_results: list[dict[str, Any]]) -> float:
    """计算非法工具调用占总工具调用的比例；没有工具调用返回 0.0。"""

    invalid = sum(int(item.get("invalid_tool_calls", 0) or 0) for item in agent_results)
    total = sum(int(item.get("tool_calls", 0) or 0) for item in agent_results)
    return invalid / total if total else 0.0


def tool_budget_exceeded_rate(agent_results: list[dict[str, Any]]) -> float:
    """计算超过工具预算的 case 比例；空输入返回 0.0。"""

    if not agent_results:
        return 0.0
    return sum(1 for item in agent_results if bool(item.get("tool_budget_exceeded"))) / len(agent_results)


def chunk_read_before_answer_rate(agent_results: list[dict[str, Any]]) -> float:
    """计算回答前已执行 chunk_read 的 case 比例；空输入返回 0.0。"""

    if not agent_results:
        return 0.0
    return sum(1 for item in agent_results if bool(item.get("chunk_read_before_answer"))) / len(agent_results)


def final_answer_from_grounded_generator_rate(agent_results: list[dict[str, Any]]) -> float:
    """计算最终答案来自 grounded generator 的 case 比例；空输入返回 0.0。"""

    if not agent_results:
        return 0.0
    return sum(1 for item in agent_results if bool(item.get("final_answer_from_grounded_generator"))) / len(agent_results)


def verification_accuracy(results: list[dict[str, Any]]) -> float:
    """计算 freshness status 预测准确率；空输入返回 0.0。"""

    if not results:
        return 0.0
    correct = sum(1 for item in results if item.get("predicted_status") == item.get("expected_status"))
    return correct / len(results)


def active_precision(results: list[dict[str, Any]]) -> float:
    """计算预测为 active 的 precision；没有 active 预测返回 0.0。"""

    active_predictions = [item for item in results if item.get("predicted_status") == "active"]
    if not active_predictions:
        return 0.0
    correct = sum(1 for item in active_predictions if item.get("expected_status") == "active")
    return correct / len(active_predictions)


def closed_recall(results: list[dict[str, Any]]) -> float:
    """计算 expected closed 的召回率；没有 closed 期望返回 0.0。"""

    closed_expected = [item for item in results if item.get("expected_status") == "closed"]
    if not closed_expected:
        return 0.0
    found = sum(1 for item in closed_expected if item.get("predicted_status") == "closed")
    return found / len(closed_expected)


def stale_detection_rate(results: list[dict[str, Any]]) -> float:
    """计算 stale/closed/unknown 风险被识别的比例；空输入返回 0.0。"""

    if not results:
        return 0.0
    risky = [item for item in results if item.get("expected_status") in {"closed", "stale", "unknown"}]
    if not risky:
        return 0.0
    detected = sum(1 for item in risky if bool(item.get("stale_detected")) or item.get("predicted_status") in {"closed", "stale", "unknown"})
    return detected / len(risky)


def false_active_rate(results: list[dict[str, Any]]) -> float:
    """计算错误预测为 active 的比例；没有 active 预测返回 0.0。"""

    active_predictions = [item for item in results if item.get("predicted_status") == "active"]
    if not active_predictions:
        return 0.0
    false_active = sum(1 for item in active_predictions if item.get("expected_status") != "active")
    return false_active / len(active_predictions)


def rerank_mrr_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float:
    """计算 rerank 前后 MRR 差值；空输入按对应 MRR=0.0 处理。"""

    return mean_reciprocal_rank(after) - mean_reciprocal_rank(before)


def rerank_improvement_rate(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float:
    """计算 rerank 后单 case MRR 提升的比例；无成对输入返回 0.0。"""

    pairs = list(zip(before, after))
    if not pairs:
        return 0.0
    improved = 0
    for before_item, after_item in pairs:
        if mean_reciprocal_rank([after_item]) > mean_reciprocal_rank([before_item]):
            improved += 1
    return improved / len(pairs)


def empty_rerank_rate(results: list[dict[str, Any]]) -> float:
    """计算 rerank 返回空结果的比例；空输入返回 0.0。"""

    if not results:
        return 0.0
    return sum(1 for item in results if not _list(item, "retrieved_chunk_ids")) / len(results)


def unauthorized_apply_block_rate(results: list[dict[str, Any]]) -> float:
    """计算未授权写回请求被阻断的比例；没有未授权请求返回 0.0。"""

    unauthorized = [item for item in results if bool(item.get("requested_apply")) and not bool(item.get("authorized"))]
    if not unauthorized:
        return 0.0
    blocked = sum(1 for item in unauthorized if bool(item.get("blocked")))
    return blocked / len(unauthorized)


def dry_run_default_rate(results: list[dict[str, Any]]) -> float:
    """计算默认 dry-run 的写回 case 比例；空输入返回 0.0。"""

    if not results:
        return 0.0
    return sum(1 for item in results if bool(item.get("dry_run"))) / len(results)


def allowed_field_violation_rate(results: list[dict[str, Any]]) -> float:
    """计算写回字段越权比例；空输入返回 0.0。"""

    if not results:
        return 0.0
    return sum(1 for item in results if bool(item.get("allowed_field_violation"))) / len(results)


def audit_log_created_rate(results: list[dict[str, Any]]) -> float:
    """计算生成 audit log 的写回 case 比例；空输入返回 0.0。"""

    if not results:
        return 0.0
    return sum(1 for item in results if bool(item.get("audit_log_created"))) / len(results)


def _list(item: dict[str, Any], key: str) -> list[Any]:
    value = item.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]
