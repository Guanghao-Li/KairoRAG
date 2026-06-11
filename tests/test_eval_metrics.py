from kairorag.evaluation.metrics import (
    evidence_recall,
    groundedness_score,
    job_verification_metrics,
    retrieval_hit_rate,
    skill_extraction_scores,
)


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

