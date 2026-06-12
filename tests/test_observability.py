import json
from dataclasses import dataclass

from kairorag.cloud.observability import TraceEvent, TraceLogger, now_iso, redact_secrets
from kairorag.cloud.query_cli import _write_trace_output


@dataclass(frozen=True)
class FakeTraceStep:
    step: str
    detail: dict


class FakeResult:
    query = "RAG"
    trace = [FakeTraceStep("rerank", {"chunk_ids": ["c1"]})]
    metrics = {"rerank_output_count": 1}


def test_trace_logger_writes_jsonl_and_creates_directory(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    logger = TraceLogger(str(path), enabled=True)

    logger.log(TraceEvent("retrieval_started", now_iso(), "run-1", "RAG", {"ok": True}))

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["event_type"] == "retrieval_started"


def test_redact_secrets_hides_sensitive_keys_and_values():
    data = {
        "api_key": "abc",
        "nested": {"authorization": "Bearer x", "note": "sk-secret-value"},
        "safe": "hello",
    }

    redacted = redact_secrets(data)

    assert redacted["api_key"] == "[已隐藏]"
    assert redacted["nested"]["authorization"] == "[已隐藏]"
    assert redacted["nested"]["note"] == "[已隐藏]"
    assert redacted["safe"] == "hello"


def test_cli_trace_output_file_generation(tmp_path):
    output = tmp_path / "run_trace.json"

    _write_trace_output(str(output), FakeResult())

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query"] == "RAG"
    assert payload["trace"][0]["step"] == "rerank"
