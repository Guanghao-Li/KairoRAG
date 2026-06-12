from pydantic import SecretStr

from kairorag.cloud import doctor as doctor_module
from kairorag.cloud.doctor import CloudDoctor, report_to_json, settings_summary
from kairorag.config import KairoCloudSettings


def _settings(tmp_path, **overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
        "cloud_index_manifest_path": str(tmp_path / "manifest.json"),
        "cloud_bm25_index_path": str(tmp_path / "bm25.json"),
        "freshness_audit_log_path": str(tmp_path / "audit" / "freshness.jsonl"),
        "trace_log_path": str(tmp_path / "trace" / "trace.jsonl"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def test_doctor_missing_config_fails(tmp_path):
    report = CloudDoctor(_settings(tmp_path, openai_api_key=None)).run(checks=["config"])

    assert report.ok is False
    assert report.checks[0].name == "config"
    assert "OPENAI_API_KEY" in report.checks[0].message


def test_doctor_complete_config_ok(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bm25.json").write_text("[]", encoding="utf-8")

    report = CloudDoctor(_settings(tmp_path)).run(live=False)

    assert report.ok is True
    assert {item.name for item in report.checks} >= {"config", "paths", "qdrant", "openai"}


def test_doctor_json_output_and_secret_redaction(tmp_path):
    report = CloudDoctor(_settings(tmp_path)).run(checks=["openai"])
    payload = report_to_json(report)
    summary = settings_summary(_settings(tmp_path), show_paths=True)

    assert "placeholder-openai" not in payload
    assert "placeholder-openai" not in str(summary)
    assert "has_openai_api_key" in summary


def test_doctor_live_false_does_not_call_external_provider(monkeypatch, tmp_path):
    def fail_if_called(settings):
        raise AssertionError("live=false 不应初始化 Qdrant provider")

    monkeypatch.setattr(doctor_module, "build_vector_store_provider", fail_if_called)

    report = CloudDoctor(_settings(tmp_path)).run(live=False, checks=["qdrant"])

    assert report.ok is True
    assert "未执行 live healthcheck" in report.checks[0].message
