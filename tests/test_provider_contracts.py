import pytest
from pydantic import SecretStr

from kairorag.config import KairoCloudSettings
from kairorag.providers import (
    BM25KeywordSearchProvider,
    BaseScoreRerankerProvider,
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
    QdrantVectorStoreProvider,
    TavilyWebSearchProvider,
    build_embedding_provider,
    build_keyword_search_provider,
    build_llm_provider,
    build_reranker_provider,
    build_vector_store_provider,
    build_web_search_provider,
)
from kairorag.providers.errors import KairoProviderError


def _settings(**overrides):
    values = {
        "openai_api_key": SecretStr("placeholder-openai"),
        "qdrant_url": "https://qdrant.example.invalid",
        "qdrant_api_key": SecretStr("placeholder-qdrant"),
        "tavily_api_key": SecretStr("placeholder-tavily"),
    }
    values.update(overrides)
    return KairoCloudSettings(**values)


def test_provider_factory_rejects_unsupported_provider():
    with pytest.raises(KairoProviderError) as exc_info:
        build_llm_provider(_settings(llm_provider="local"))

    assert "不支持" in str(exc_info.value)


def test_provider_factory_rejects_missing_key():
    with pytest.raises(KairoProviderError) as exc_info:
        build_web_search_provider(_settings(tavily_api_key=None))

    assert "缺少" in str(exc_info.value)


def test_provider_factory_returns_expected_types():
    settings = _settings()

    assert isinstance(build_llm_provider(settings), OpenAILLMProvider)
    assert isinstance(build_embedding_provider(settings), OpenAIEmbeddingProvider)
    assert isinstance(build_vector_store_provider(settings), QdrantVectorStoreProvider)
    assert isinstance(build_keyword_search_provider(settings), BM25KeywordSearchProvider)
    assert isinstance(build_web_search_provider(settings), TavilyWebSearchProvider)
    assert isinstance(build_reranker_provider(settings), BaseScoreRerankerProvider)


def test_cross_encoder_skeleton_requires_model_config():
    with pytest.raises(KairoProviderError) as exc_info:
        build_reranker_provider(_settings(reranker_provider="cross_encoder"))

    assert "CROSS_ENCODER_MODEL" in str(exc_info.value)
