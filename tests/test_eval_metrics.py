from kairorag.evaluation.metrics import (
    evidence_recall,
    groundedness_score,
    job_verification_metrics,
    retrieval_hit_rate,
    skill_extraction_scores,
)
from kairorag.eval import metrics as cloud_metrics


def test_rag_metrics_are_deterministic():
    assert retrieval_hit_rate(["a"], ["b", "a"], []) == 1.0
    assert evidence_recall(["a", "b"], ["a"]) == 0.5
    assert groundedness_score(["a"], ["a"], ["a", "b"]) == 1.0
    scores = skill_extraction_scores(["RAG", "Python"], ["rag", "LangGraph"])
    assert round(scores["precision"], 2) == 0.5
    assert round(scores["recall"], 2) == 0.5
    assert round(scores["f1"], 2) == 0.5


def test_job_verification_metrics():
    expected = [
        {"job_id": "j1", "expected_status": "active", "expected_needs_reindex": False, "expected_evidence_domain": "mock"},
        {"job_id": "j2", "expected_status": "closed", "expected_needs_reindex": True, "expected_evidence_domain": "mock"},
    ]
    predictions = [
        {"job_id": "j1", "new_status": "active", "needs_reindex": False, "evidence": [{"url": "mock://a"}]},
        {"job_id": "j2", "new_status": "closed", "needs_reindex": True, "evidence": [{"url": "mock://b"}]},
    ]
    metrics = job_verification_metrics(expected, predictions)
    assert metrics["verification_accuracy"] == 1.0
    assert metrics["closed_recall"] == 1.0


def test_cloud_eval_retrieval_metrics_and_empty_inputs():
    results = [
        {"expected_chunk_ids": ["a"], "retrieved_chunk_ids": ["x", "a"], "read_chunk_ids": ["a"]},
        {"expected_chunk_ids": ["b"], "retrieved_chunk_ids": ["b"], "read_chunk_ids": []},
    ]

    assert cloud_metrics.retrieval_hit_rate(results) == 1.0
    assert cloud_metrics.evidence_recall(results) == 0.5
    assert round(cloud_metrics.mean_reciprocal_rank(results), 2) == 0.75
    assert round(cloud_metrics.ndcg_at_k(results, k=2), 2) == 0.82
    assert cloud_metrics.retrieval_hit_rate([]) == 0.0


def test_cloud_eval_groundedness_and_safety_metrics():
    answers = [
        {"citations": ["a"], "read_chunk_ids": ["a"], "used_search_snippet_as_evidence": False},
        {"citations": ["b"], "read_chunk_ids": ["a"], "used_search_snippet_as_evidence": True},
    ]

    assert cloud_metrics.citation_validity(answers[0]) == 1.0
    assert cloud_metrics.citation_validity(answers[1]) == 0.0
    assert cloud_metrics.invalid_citation_rate(answers) == 0.5
    assert cloud_metrics.snippet_leakage_rate(answers) == 0.5


def test_cloud_eval_freshness_rerank_and_writeback_metrics():
    freshness = [
        {"expected_status": "active", "predicted_status": "active"},
        {"expected_status": "closed", "predicted_status": "active"},
    ]
    before = [{"expected_chunk_ids": ["a"], "retrieved_chunk_ids": ["x", "a"]}]
    after = [{"expected_chunk_ids": ["a"], "retrieved_chunk_ids": ["a", "x"]}]
    writeback = [
        {"requested_apply": True, "authorized": False, "blocked": True, "dry_run": True},
        {"requested_apply": False, "authorized": True, "blocked": False, "dry_run": False},
    ]

    assert cloud_metrics.verification_accuracy(freshness) == 0.5
    assert cloud_metrics.false_active_rate(freshness) == 0.5
    assert cloud_metrics.rerank_mrr_delta(before, after) > 0
    assert cloud_metrics.unauthorized_apply_block_rate(writeback) == 1.0
