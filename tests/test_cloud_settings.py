import pytest
from pydantic import SecretStr

from kairorag.config import CloudRuntimeConfigurationError, KairoCloudSettings, validate_cloud_runtime


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def test_validate_cloud_runtime_missing_openai_key():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(openai_api_key=None))

    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_validate_cloud_runtime_missing_qdrant_settings():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(qdrant_url=None, qdrant_api_key=None))

    message = str(exc_info.value)
    assert "QDRANT_URL" in message
    assert "QDRANT_API_KEY" in message


def test_validate_cloud_runtime_complete_settings_pass():
    checked = validate_cloud_runtime(_settings())

    assert checked.llm_provider == "openai"
    assert checked.vector_store_provider == "qdrant"
    assert checked.web_search_provider == "tavily"
