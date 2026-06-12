"""Cloud-native provider 工厂。"""

from __future__ import annotations

from kairorag.config import KairoCloudSettings
from kairorag.providers.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.keyword import BM25KeywordSearchProvider, KeywordSearchProvider
from kairorag.providers.llm import LLMProvider, OpenAILLMProvider
from kairorag.providers.rerankers import (
    BaseScoreRerankerProvider,
    CrossEncoderRerankerProvider,
    RerankerProvider,
)
from kairorag.providers.vectorstores import QdrantVectorStoreProvider, VectorStoreProvider
from kairorag.providers.websearch import (
    BingWebSearchProvider,
    SerpAPIWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
)


def build_llm_provider(settings: KairoCloudSettings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(settings)
    raise KairoProviderError(f"不支持的 LLM_PROVIDER：{settings.llm_provider}")


def build_embedding_provider(settings: KairoCloudSettings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    raise KairoProviderError(f"不支持的 EMBEDDING_PROVIDER：{settings.embedding_provider}")


def build_vector_store_provider(settings: KairoCloudSettings) -> VectorStoreProvider:
    if settings.vector_store_provider == "qdrant":
        return QdrantVectorStoreProvider(settings)
    raise KairoProviderError(f"不支持的 VECTOR_STORE_PROVIDER：{settings.vector_store_provider}")


def build_keyword_search_provider(settings: KairoCloudSettings) -> KeywordSearchProvider:
    del settings
    return BM25KeywordSearchProvider()


def build_web_search_provider(settings: KairoCloudSettings) -> WebSearchProvider:
    if settings.web_search_provider == "tavily":
        return TavilyWebSearchProvider(settings)
    if settings.web_search_provider == "serpapi":
        return SerpAPIWebSearchProvider(settings)
    if settings.web_search_provider == "bing":
        return BingWebSearchProvider(settings)
    raise KairoProviderError(f"不支持的 WEB_SEARCH_PROVIDER：{settings.web_search_provider}")


def build_reranker_provider(settings: KairoCloudSettings) -> RerankerProvider:
    if settings.reranker_provider == "bm25":
        return BaseScoreRerankerProvider()
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderRerankerProvider(settings)
    raise KairoProviderError(f"不支持的 RERANKER_PROVIDER：{settings.reranker_provider}")


__all__ = [
    "BM25KeywordSearchProvider",
    "BingWebSearchProvider",
    "CrossEncoderRerankerProvider",
    "KairoProviderError",
    "OpenAIEmbeddingProvider",
    "OpenAILLMProvider",
    "QdrantVectorStoreProvider",
    "SerpAPIWebSearchProvider",
    "TavilyWebSearchProvider",
    "build_embedding_provider",
    "build_keyword_search_provider",
    "build_llm_provider",
    "build_reranker_provider",
    "build_vector_store_provider",
    "build_web_search_provider",
]
