import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers import build_reranker_provider
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.rerankers import (
    BaseScoreRerankerProvider,
    CohereRerankProvider,
    CrossEncoderRerankerProvider,
    JinaRerankProvider,
    OpenAIListwiseRerankProvider,
    VoyageRerankProvider,
)


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "cohere_api_key": SecretStr("placeholder-cohere"),
        "jina_api_key": SecretStr("placeholder-jina"),
        "voyage_api_key": SecretStr("placeholder-voyage"),
        "cross_encoder_model": "fake-model",
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def test_reranker_factory_covers_cloud_and_baseline(monkeypatch):
    monkeypatch.setattr(CrossEncoderRerankerProvider, "_load_model", lambda self, model_name: object())

    cases = [
        ("base_score", BaseScoreRerankerProvider),
        ("cohere", CohereRerankProvider),
        ("jina", JinaRerankProvider),
        ("voyage", VoyageRerankProvider),
        ("openai_listwise", OpenAIListwiseRerankProvider),
        ("cross_encoder", CrossEncoderRerankerProvider),
    ]
    for provider_name, expected_type in cases:
        provider = build_reranker_provider(_settings(reranker_provider=provider_name))
        assert isinstance(provider, expected_type)


def test_reranker_factory_missing_key_raises_chinese_error():
    cases = [
        ("cohere", {"cohere_api_key": None}, "COHERE_API_KEY"),
        ("jina", {"jina_api_key": None}, "JINA_API_KEY"),
        ("voyage", {"voyage_api_key": None}, "VOYAGE_API_KEY"),
        ("openai_listwise", {"openai_api_key": None}, "OPENAI_API_KEY"),
        ("cross_encoder", {"cross_encoder_model": None}, "CROSS_ENCODER_MODEL"),
    ]
    for provider_name, overrides, marker in cases:
        with pytest.raises(KairoProviderError) as exc_info:
            build_reranker_provider(_settings(reranker_provider=provider_name, **overrides))
        assert marker in str(exc_info.value)


def test_reranker_factory_rejects_bm25():
    with pytest.raises(KairoProviderError) as exc_info:
        build_reranker_provider(_settings(reranker_provider="bm25"))

    assert "RERANKER_PROVIDER" in str(exc_info.value)
