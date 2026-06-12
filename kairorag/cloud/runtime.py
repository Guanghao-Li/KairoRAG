"""Cloud-native runtime 组装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kairorag.cloud.agent import CloudLLMAgent
from kairorag.cloud.answering import GroundedCloudAnswerGenerator
from kairorag.cloud.freshness import CloudFreshnessUpdater, CloudJobFreshnessVerifier
from kairorag.cloud.manifest import CloudIndexManifest, load_manifest
from kairorag.cloud.query import CloudQueryService
from kairorag.cloud.retriever import CloudRetriever
from kairorag.config import KairoCloudSettings
from kairorag.config.settings import CloudRuntimeConfigurationError, validate_cloud_runtime
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
    llm_provider: LLMProvider
    embedding_provider: EmbeddingProvider
    vector_store: VectorStoreProvider
    keyword_search: KeywordSearchProvider
    reranker: RerankerProvider

    @property
    def llm(self) -> LLMProvider:
        """兼容阶段一代码的别名。"""

        return self.llm_provider

    @property
    def embeddings(self) -> EmbeddingProvider:
        """兼容阶段一代码的别名。"""

        return self.embedding_provider


@dataclass(frozen=True)
class CloudQueryRuntime(CloudRuntime):
    """可直接执行 cloud query 的 runtime。"""

    manifest: CloudIndexManifest
    retriever: CloudRetriever
    answer_generator: GroundedCloudAnswerGenerator
    query_service: CloudQueryService


@dataclass(frozen=True)
class CloudAgentRuntime(CloudQueryRuntime):
    """可直接执行 LLM autonomous agent 的 runtime。"""

    llm_agent: CloudLLMAgent
    web_search_provider: WebSearchProvider | None
    freshness_verifier: CloudJobFreshnessVerifier | None
    freshness_updater: CloudFreshnessUpdater | None


def build_cloud_runtime(settings: KairoCloudSettings | None = None) -> CloudRuntime:
    """校验配置并构建真实 provider，不 fallback 到 legacy 本地实现。"""

    checked = validate_cloud_runtime(settings)
    return CloudRuntime(
        settings=checked,
        llm_provider=build_llm_provider(checked),
        embedding_provider=build_embedding_provider(checked),
        vector_store=build_vector_store_provider(checked),
        keyword_search=build_keyword_search_provider(checked),
        reranker=build_reranker_provider(checked),
    )


def build_cloud_query_runtime(settings: KairoCloudSettings | None = None) -> CloudQueryRuntime:
    """构建 query runtime，并加载 manifest 与 BM25 keyword index。"""

    runtime = build_cloud_runtime(settings)
    manifest_path = Path(runtime.settings.cloud_index_manifest_path)
    if not manifest_path.exists():
        raise CloudRuntimeConfigurationError(
            "缺少 cloud index manifest："
            f"{manifest_path}。请先运行：python -m kairorag.cloud.index"
        )
    bm25_path = Path(runtime.settings.cloud_bm25_index_path)
    if not bm25_path.exists():
        raise CloudRuntimeConfigurationError(
            "缺少 cloud BM25 index："
            f"{bm25_path}。请先运行：python -m kairorag.cloud.index"
        )

    manifest = load_manifest(manifest_path)
    if not hasattr(runtime.keyword_search, "load_keyword_documents"):
        raise CloudRuntimeConfigurationError("当前 keyword search provider 不支持加载 cloud BM25 index。")
    runtime.keyword_search.load_keyword_documents(bm25_path)
    retriever = CloudRetriever(
        runtime.settings,
        runtime.embedding_provider,
        runtime.vector_store,
        runtime.keyword_search,
        runtime.reranker,
        manifest,
    )
    answer_generator = GroundedCloudAnswerGenerator(runtime.settings, runtime.llm_provider)
    query_service = CloudQueryService(runtime.settings, retriever, answer_generator)
    return CloudQueryRuntime(
        settings=runtime.settings,
        llm_provider=runtime.llm_provider,
        embedding_provider=runtime.embedding_provider,
        vector_store=runtime.vector_store,
        keyword_search=runtime.keyword_search,
        reranker=runtime.reranker,
        manifest=manifest,
        retriever=retriever,
        answer_generator=answer_generator,
        query_service=query_service,
    )


def build_cloud_agent_runtime(
    settings: KairoCloudSettings | None = None,
    *,
    freshness_apply_authorized: bool = False,
) -> CloudAgentRuntime:
    """构建 cloud Agent runtime，不 fallback 到 legacy。"""

    query_runtime = build_cloud_query_runtime(settings)
    web_search_provider = None
    freshness_verifier = None
    freshness_updater = None
    if query_runtime.settings.job_freshness_enabled:
        web_search_provider = build_web_search_provider(query_runtime.settings)
        freshness_verifier = CloudJobFreshnessVerifier(query_runtime.settings, web_search_provider)
        freshness_updater = CloudFreshnessUpdater(query_runtime.settings)
    llm_agent = CloudLLMAgent(
        settings=query_runtime.settings,
        retriever=query_runtime.retriever,
        answer_generator=query_runtime.answer_generator,
        llm_provider=query_runtime.llm_provider,
        web_search_provider=web_search_provider,
        freshness_verifier=freshness_verifier,
        freshness_updater=freshness_updater,
        vector_store=query_runtime.vector_store,
        freshness_apply_authorized=freshness_apply_authorized,
    )
    return CloudAgentRuntime(
        settings=query_runtime.settings,
        llm_provider=query_runtime.llm_provider,
        embedding_provider=query_runtime.embedding_provider,
        vector_store=query_runtime.vector_store,
        keyword_search=query_runtime.keyword_search,
        reranker=query_runtime.reranker,
        manifest=query_runtime.manifest,
        retriever=query_runtime.retriever,
        answer_generator=query_runtime.answer_generator,
        query_service=query_runtime.query_service,
        llm_agent=llm_agent,
        web_search_provider=web_search_provider,
        freshness_verifier=freshness_verifier,
        freshness_updater=freshness_updater,
    )
