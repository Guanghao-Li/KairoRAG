"""Cloud-native runtime 配置。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudRuntimeConfigurationError(RuntimeError):
    """cloud runtime 配置缺失或 provider 名称不受支持时抛出的错误。"""


class KairoCloudSettings(BaseSettings):
    """KairoRAG cloud 主路径的统一配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    kairo_env: Literal["dev", "staging", "prod"] = Field(default="dev", validation_alias="KAIRO_ENV")
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="openai", validation_alias="EMBEDDING_PROVIDER")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )

    vector_store_provider: str = Field(default="qdrant", validation_alias="VECTOR_STORE_PROVIDER")
    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="kairo_chunks", validation_alias="QDRANT_COLLECTION")

    web_search_provider: str = Field(default="tavily", validation_alias="WEB_SEARCH_PROVIDER")
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    serpapi_api_key: SecretStr | None = Field(default=None, validation_alias="SERPAPI_API_KEY")
    bing_api_key: SecretStr | None = Field(default=None, validation_alias="BING_API_KEY")

    reranker_provider: str = Field(default="bm25", validation_alias="RERANKER_PROVIDER")
    cross_encoder_model: str | None = Field(default=None, validation_alias="CROSS_ENCODER_MODEL")

    max_search_results: int = Field(default=10, validation_alias="MAX_SEARCH_RESULTS")
    max_chunks_to_read: int = Field(default=5, validation_alias="MAX_CHUNKS_TO_READ")
    max_context_tokens: int = Field(default=6000, validation_alias="MAX_CONTEXT_TOKENS")
    max_tool_calls: int = Field(default=20, validation_alias="MAX_TOOL_CALLS")
    request_timeout_seconds: int = Field(default=30, validation_alias="REQUEST_TIMEOUT_SECONDS")

    @field_validator(
        "llm_provider",
        "embedding_provider",
        "vector_store_provider",
        "web_search_provider",
        "reranker_provider",
        mode="before",
    )
    @classmethod
    def _normalize_provider_name(cls, value: str) -> str:
        return str(value).strip().lower()


def _has_secret(value: SecretStr | None) -> bool:
    return bool(value and value.get_secret_value().strip())


def validate_cloud_runtime(settings: KairoCloudSettings | None = None) -> KairoCloudSettings:
    """校验 cloud 主路径必需的 provider 和密钥配置。"""

    settings = settings or KairoCloudSettings()
    missing: list[str] = []
    unsupported: list[str] = []

    if settings.llm_provider != "openai":
        unsupported.append(f"LLM_PROVIDER={settings.llm_provider}")
    if settings.embedding_provider != "openai":
        unsupported.append(f"EMBEDDING_PROVIDER={settings.embedding_provider}")
    if settings.vector_store_provider != "qdrant":
        unsupported.append(f"VECTOR_STORE_PROVIDER={settings.vector_store_provider}")
    if settings.web_search_provider not in {"tavily", "serpapi", "bing"}:
        unsupported.append(f"WEB_SEARCH_PROVIDER={settings.web_search_provider}")
    if settings.reranker_provider not in {"bm25", "cross_encoder"}:
        unsupported.append(f"RERANKER_PROVIDER={settings.reranker_provider}")

    if settings.llm_provider == "openai" and not _has_secret(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if settings.embedding_provider == "openai" and not _has_secret(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if settings.vector_store_provider == "qdrant":
        if not settings.qdrant_url:
            missing.append("QDRANT_URL")
        if not _has_secret(settings.qdrant_api_key):
            missing.append("QDRANT_API_KEY")
    if settings.web_search_provider == "tavily" and not _has_secret(settings.tavily_api_key):
        missing.append("TAVILY_API_KEY")
    if settings.web_search_provider == "serpapi" and not _has_secret(settings.serpapi_api_key):
        missing.append("SERPAPI_API_KEY")
    if settings.web_search_provider == "bing" and not _has_secret(settings.bing_api_key):
        missing.append("BING_API_KEY")

    problems: list[str] = []
    if unsupported:
        problems.append("不受支持的 provider：" + "、".join(unsupported))
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        problems.append("缺少必要环境变量：" + "、".join(unique_missing))

    if problems:
        raise CloudRuntimeConfigurationError("cloud runtime 配置无效；" + "；".join(problems))
    return settings
