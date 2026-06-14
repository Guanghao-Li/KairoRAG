"""Cloud-native runtime 配置。"""

from __future__ import annotations

from typing import Any, Literal

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
    web_search_timeout_seconds: int = Field(default=20, validation_alias="WEB_SEARCH_TIMEOUT_SECONDS")
    web_search_max_results: int = Field(default=5, validation_alias="WEB_SEARCH_MAX_RESULTS")
    job_verification_max_results: int = Field(default=5, validation_alias="JOB_VERIFICATION_MAX_RESULTS")
    job_verification_request_timeout_seconds: int = Field(
        default=20,
        validation_alias="JOB_VERIFICATION_REQUEST_TIMEOUT_SECONDS",
    )
    job_freshness_enabled: bool = Field(default=True, validation_alias="JOB_FRESHNESS_ENABLED")

    keyword_search_provider: str = Field(default="bm25", validation_alias="KEYWORD_SEARCH_PROVIDER")
    reranker_provider: str = Field(default="base_score", validation_alias="RERANKER_PROVIDER")
    cohere_api_key: SecretStr | None = Field(default=None, validation_alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(default="rerank-v3.5", validation_alias="COHERE_RERANK_MODEL")
    jina_api_key: SecretStr | None = Field(default=None, validation_alias="JINA_API_KEY")
    jina_rerank_model: str = Field(
        default="jina-reranker-v2-base-multilingual",
        validation_alias="JINA_RERANK_MODEL",
    )
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    voyage_rerank_model: str = Field(default="rerank-2", validation_alias="VOYAGE_RERANK_MODEL")
    openai_rerank_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_RERANK_MODEL")
    cross_encoder_model: str | None = Field(default=None, validation_alias="CROSS_ENCODER_MODEL")
    rerank_top_k: int = Field(default=5, validation_alias="RERANK_TOP_K")
    rerank_candidate_count: int = Field(default=20, validation_alias="RERANK_CANDIDATE_COUNT")
    rerank_timeout_seconds: int = Field(default=30, validation_alias="RERANK_TIMEOUT_SECONDS")
    qdrant_metadata_write_enabled: bool = Field(default=False, validation_alias="QDRANT_METADATA_WRITE_ENABLED")
    qdrant_metadata_write_dry_run: bool = Field(default=True, validation_alias="QDRANT_METADATA_WRITE_DRY_RUN")
    approval_policy: str = Field(default="require_confirmation", validation_alias="APPROVAL_POLICY")
    require_human_approval_for_writeback: bool = Field(
        default=True,
        validation_alias="REQUIRE_HUMAN_APPROVAL_FOR_WRITEBACK",
    )
    qdrant_metadata_allowed_fields: list[str] = Field(
        default_factory=lambda: [
            "verification_status",
            "archived",
            "archived_at",
            "closed_reason",
            "canonical_url",
            "original_url",
            "last_verified_at",
            "verification_confidence",
            "manual_review_required",
            "duplicate_of",
        ],
        validation_alias="QDRANT_METADATA_ALLOWED_FIELDS",
    )
    freshness_audit_log_path: str = Field(
        default="data/freshness_audit.jsonl",
        validation_alias="FRESHNESS_AUDIT_LOG_PATH",
    )
    trace_log_path: str = Field(default="data/trace_events.jsonl", validation_alias="TRACE_LOG_PATH")
    observability_enabled: bool = Field(default=True, validation_alias="OBSERVABILITY_ENABLED")

    qdrant_distance: str = Field(default="Cosine", validation_alias="QDRANT_DISTANCE")
    cloud_index_manifest_path: str = Field(
        default="data/cloud_index_manifest.json",
        validation_alias="CLOUD_INDEX_MANIFEST_PATH",
    )
    cloud_bm25_index_path: str = Field(
        default="data/cloud_bm25_index.json",
        validation_alias="CLOUD_BM25_INDEX_PATH",
    )
    cloud_chunk_batch_size: int = Field(default=64, validation_alias="CLOUD_CHUNK_BATCH_SIZE")
    cloud_recreate_collection: bool = Field(default=False, validation_alias="CLOUD_RECREATE_COLLECTION")

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
        "keyword_search_provider",
        "reranker_provider",
        "approval_policy",
        mode="before",
    )
    @classmethod
    def _normalize_provider_name(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator("qdrant_distance", mode="before")
    @classmethod
    def _normalize_qdrant_distance(cls, value: str) -> str:
        return str(value).strip() or "Cosine"

    @field_validator("qdrant_metadata_allowed_fields", mode="before")
    @classmethod
    def _normalize_allowed_fields(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("QDRANT_METADATA_ALLOWED_FIELDS 必须是逗号分隔字符串或列表。")


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
    if settings.keyword_search_provider != "bm25":
        unsupported.append(f"KEYWORD_SEARCH_PROVIDER={settings.keyword_search_provider}")
    supported_rerankers = {"base_score", "cohere", "jina", "voyage", "openai_listwise", "cross_encoder"}
    if settings.reranker_provider not in supported_rerankers:
        unsupported.append(f"RERANKER_PROVIDER={settings.reranker_provider}")
    if settings.approval_policy not in {"deny", "dry_run", "require_confirmation", "allow"}:
        unsupported.append(f"APPROVAL_POLICY={settings.approval_policy}")

    if settings.llm_provider == "openai" and not _has_secret(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if settings.embedding_provider == "openai" and not _has_secret(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if settings.vector_store_provider == "qdrant":
        if not settings.qdrant_url:
            missing.append("QDRANT_URL")
        if not _has_secret(settings.qdrant_api_key):
            missing.append("QDRANT_API_KEY")
    if settings.qdrant_metadata_write_enabled:
        if not settings.qdrant_url:
            missing.append("QDRANT_URL")
        if not _has_secret(settings.qdrant_api_key):
            missing.append("QDRANT_API_KEY")
    if settings.reranker_provider == "cohere" and not _has_secret(settings.cohere_api_key):
        missing.append("COHERE_API_KEY")
    if settings.reranker_provider == "jina" and not _has_secret(settings.jina_api_key):
        missing.append("JINA_API_KEY")
    if settings.reranker_provider == "voyage" and not _has_secret(settings.voyage_api_key):
        missing.append("VOYAGE_API_KEY")
    if settings.reranker_provider == "openai_listwise" and not _has_secret(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if settings.reranker_provider == "cross_encoder" and not settings.cross_encoder_model:
        missing.append("CROSS_ENCODER_MODEL")
    if settings.job_freshness_enabled and settings.web_search_provider in {"tavily", "serpapi", "bing"}:
        if settings.web_search_provider == "tavily" and not _has_secret(settings.tavily_api_key):
            missing.append("TAVILY_API_KEY")
        if settings.web_search_provider == "serpapi" and not _has_secret(settings.serpapi_api_key):
            missing.append("SERPAPI_API_KEY")
        if settings.web_search_provider == "bing" and not _has_secret(settings.bing_api_key):
            missing.append("BING_API_KEY")
    if settings.cloud_chunk_batch_size <= 0:
        unsupported.append(f"CLOUD_CHUNK_BATCH_SIZE={settings.cloud_chunk_batch_size}")
    if settings.web_search_timeout_seconds <= 0:
        unsupported.append(f"WEB_SEARCH_TIMEOUT_SECONDS={settings.web_search_timeout_seconds}")
    if settings.web_search_max_results <= 0:
        unsupported.append(f"WEB_SEARCH_MAX_RESULTS={settings.web_search_max_results}")
    if settings.job_verification_max_results <= 0:
        unsupported.append(f"JOB_VERIFICATION_MAX_RESULTS={settings.job_verification_max_results}")
    if settings.job_verification_request_timeout_seconds <= 0:
        unsupported.append(
            f"JOB_VERIFICATION_REQUEST_TIMEOUT_SECONDS={settings.job_verification_request_timeout_seconds}"
        )
    if settings.rerank_top_k <= 0:
        unsupported.append(f"RERANK_TOP_K={settings.rerank_top_k}")
    if settings.rerank_candidate_count <= 0:
        unsupported.append(f"RERANK_CANDIDATE_COUNT={settings.rerank_candidate_count}")
    if settings.rerank_timeout_seconds <= 0:
        unsupported.append(f"RERANK_TIMEOUT_SECONDS={settings.rerank_timeout_seconds}")

    problems: list[str] = []
    if unsupported:
        problems.append("不受支持的 provider：" + "、".join(unsupported))
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        problems.append("缺少必要环境变量：" + "、".join(unique_missing))

    if problems:
        raise CloudRuntimeConfigurationError("cloud runtime 配置无效；" + "；".join(problems))
    return settings
