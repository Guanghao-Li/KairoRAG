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
    assert checked.keyword_search_provider == "bm25"
    assert checked.reranker_provider == "base_score"


def test_validate_cloud_runtime_missing_tavily_key_when_freshness_enabled():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(tavily_api_key=None))

    assert "TAVILY_API_KEY" in str(exc_info.value)


def test_validate_cloud_runtime_allows_missing_web_key_when_freshness_disabled():
    checked = validate_cloud_runtime(_settings(tavily_api_key=None, job_freshness_enabled=False))

    assert checked.job_freshness_enabled is False


def test_validate_cloud_runtime_rejects_unsupported_web_provider():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(web_search_provider="mock"))

    assert "WEB_SEARCH_PROVIDER=mock" in str(exc_info.value)


def test_validate_cloud_runtime_rejects_unsupported_keyword_provider():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(keyword_search_provider="local"))

    assert "KEYWORD_SEARCH_PROVIDER=local" in str(exc_info.value)


def test_validate_cloud_runtime_rejects_bm25_as_reranker():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(reranker_provider="bm25"))

    assert "RERANKER_PROVIDER=bm25" in str(exc_info.value)


def test_validate_cloud_runtime_cross_encoder_requires_model_config():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(_settings(reranker_provider="cross_encoder"))

    assert "CROSS_ENCODER_MODEL" in str(exc_info.value)


def test_validate_cloud_runtime_allows_cross_encoder_with_model_config():
    checked = validate_cloud_runtime(_settings(reranker_provider="cross_encoder", cross_encoder_model="fake-model"))

    assert checked.reranker_provider == "cross_encoder"


def test_validate_cloud_runtime_reranker_provider_requires_key():
    cases = [
        ("cohere", "COHERE_API_KEY", {"cohere_api_key": None}),
        ("jina", "JINA_API_KEY", {"jina_api_key": None}),
        ("voyage", "VOYAGE_API_KEY", {"voyage_api_key": None}),
        ("openai_listwise", "OPENAI_API_KEY", {"openai_api_key": None}),
    ]
    for provider, env_name, overrides in cases:
        with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
            validate_cloud_runtime(_settings(reranker_provider=provider, **overrides))
        assert env_name in str(exc_info.value)


def test_validate_cloud_runtime_reranker_provider_with_key_passes():
    checked = validate_cloud_runtime(
        _settings(
            reranker_provider="cohere",
            cohere_api_key=SecretStr("placeholder-cohere"),
        )
    )

    assert checked.reranker_provider == "cohere"


def test_validate_cloud_runtime_metadata_write_requires_qdrant_config():
    with pytest.raises(CloudRuntimeConfigurationError) as exc_info:
        validate_cloud_runtime(
            _settings(
                qdrant_metadata_write_enabled=True,
                qdrant_url=None,
                qdrant_api_key=None,
            )
        )

    message = str(exc_info.value)
    assert "QDRANT_URL" in message
    assert "QDRANT_API_KEY" in message
