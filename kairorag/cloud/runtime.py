"""Cloud-native runtime 组装。"""

from __future__ import annotations

from dataclasses import dataclass

from kairorag.config import KairoCloudSettings
from kairorag.config.settings import validate_cloud_runtime
from kairorag.providers import (
    build_embedding_provider,
    build_keyword_search_provider,
    build_llm_provider,
    build_reranker_provider,
    build_vector_store_provider,
    build_web_search_provider,
)
from kairorag.providers.embeddings import EmbeddingProvider
from kairorag.providers.keyword import KeywordSearchProvider
from kairorag.providers.llm import LLMProvider
from kairorag.providers.rerankers import RerankerProvider
from kairorag.providers.vectorstores import VectorStoreProvider
from kairorag.providers.websearch import WebSearchProvider


@dataclass(frozen=True)
class CloudRuntime:
    """cloud 主路径所需 provider 的聚合对象。"""

    settings: KairoCloudSettings
    llm: LLMProvider
    embeddings: EmbeddingProvider
    vector_store: VectorStoreProvider
    keyword_search: KeywordSearchProvider
    web_search: WebSearchProvider
    reranker: RerankerProvider


def build_cloud_runtime(settings: KairoCloudSettings | None = None) -> CloudRuntime:
    """校验配置并构建真实 provider，不 fallback 到 legacy 本地实现。"""

    checked = validate_cloud_runtime(settings)
    return CloudRuntime(
        settings=checked,
        llm=build_llm_provider(checked),
        embeddings=build_embedding_provider(checked),
        vector_store=build_vector_store_provider(checked),
        keyword_search=build_keyword_search_provider(checked),
        web_search=build_web_search_provider(checked),
        reranker=build_reranker_provider(checked),
    )
