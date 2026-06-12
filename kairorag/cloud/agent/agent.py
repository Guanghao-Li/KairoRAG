"""Cloud LLM Autonomous Planning Agent 高层封装。"""

from __future__ import annotations

from typing import Any

from kairorag.cloud.agent.executor import LLMAgentExecutor
from kairorag.cloud.agent.planner import LLMToolPlanner
from kairorag.cloud.agent.registry import ToolRegistry
from kairorag.cloud.agent.schemas import AgentRunResult
from kairorag.cloud.answering import GroundedCloudAnswerGenerator
from kairorag.cloud.freshness import CloudFreshnessUpdater, CloudJobFreshnessVerifier
from kairorag.cloud.retriever import CloudRetriever
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMProvider
from kairorag.providers.vectorstores import VectorStoreProvider
from kairorag.providers.websearch import WebSearchProvider


class CloudLLMAgent:
    """对外暴露 ask 的 cloud-native LLM Agent。"""

    def __init__(
        self,
        settings: KairoCloudSettings,
        retriever: CloudRetriever,
        answer_generator: GroundedCloudAnswerGenerator,
        llm_provider: LLMProvider,
        web_search_provider: WebSearchProvider | None = None,
        freshness_verifier: CloudJobFreshnessVerifier | None = None,
        freshness_updater: CloudFreshnessUpdater | None = None,
        vector_store: VectorStoreProvider | None = None,
        freshness_apply_authorized: bool = False,
    ) -> None:
        self.settings = settings
        self.registry = ToolRegistry(
            settings,
            retriever,
            answer_generator,
            web_search_provider=web_search_provider,
            freshness_verifier=freshness_verifier,
            freshness_updater=freshness_updater,
            vector_store=vector_store,
            freshness_apply_authorized=freshness_apply_authorized,
        )
        self.planner = LLMToolPlanner(settings, llm_provider, self.registry)
        self.executor = LLMAgentExecutor(settings, self.planner, self.registry)

    def ask(self, query: str, *, filters: dict[str, Any] | None = None) -> AgentRunResult:
        return self.executor.run(query, filters=filters)
