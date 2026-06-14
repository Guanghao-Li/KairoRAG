from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.ui.app import create_dashboard_app


def _settings(tmp_path):
    return KairoCloudSettings(
        openai_api_key=SecretStr("sk-test-secret-value-123456"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
        tavily_api_key=SecretStr("placeholder-tavily"),
        cloud_index_manifest_path=str(tmp_path / "manifest.json"),
        cloud_bm25_index_path=str(tmp_path / "bm25.json"),
        trace_log_path=str(tmp_path / "trace.jsonl"),
        freshness_audit_log_path=str(tmp_path / "audit.jsonl"),
    )


def test_ui_app_home_and_doctor_return_200(tmp_path):
    app = create_dashboard_app(root=tmp_path, settings=_settings(tmp_path))

    home = app.handle("/")
    doctor = app.handle("/doctor")

    assert home.status == 200
    assert doctor.status == 200
    assert "KairoRAG Dashboard" in home.body


def test_ui_trace_does_not_leak_secrets(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "trace.jsonl").write_text(
        '{"OPENAI_API_KEY":"sk-test-secret-value-123456","text":"'
        + ("长文本" * 200)
        + '"}\n',
        encoding="utf-8",
    )
    app = create_dashboard_app(root=tmp_path, settings=settings)

    response = app.handle("/trace")

    assert response.status == 200
    assert "sk-test-secret-value" not in response.body
    assert "[已隐藏]" in response.body
