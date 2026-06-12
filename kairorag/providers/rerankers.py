"""Reranker provider 接口和真实 cloud reranker 实现。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from kairorag.config import KairoCloudSettings
from kairorag.providers.errors import KairoProviderError
from kairorag.providers.llm import LLMMessage, LLMProvider, OpenAILLMProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    base_score: float | None = None


@dataclass(frozen=True)
class RerankResult:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class RerankerProvider(Protocol):
    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        ...


class BaseScoreRerankerProvider:
    """baseline/debug 重排器：只按 RRF/base_score 排序，不伪装成真实 reranker。"""

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        _validate_rerank_input(query, top_k)
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda item: item.base_score or 0.0, reverse=True)
        return [
            RerankResult(
                chunk_id=candidate.chunk_id,
                score=float(candidate.base_score or 0.0),
                metadata=dict(candidate.metadata),
            )
            for candidate in ranked[:top_k]
        ]


class _HTTPRerankProvider:
    provider_name = "http"
    endpoint = ""
    top_k_field = "top_n"
    result_fields = ("results",)

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise KairoProviderError(f"无法初始化 {self.provider_name} reranker：缺少 API key。")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._client = client

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        _validate_rerank_input(query, top_k)
        if not candidates:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            self.top_k_field: top_k,
        }
        try:
            response = self._post(payload)
            data = response.json()
        except httpx.TimeoutException as exc:
            raise KairoProviderError(f"{self.provider_name} rerank 请求超时。") from exc
        except httpx.HTTPError as exc:
            raise KairoProviderError(f"{self.provider_name} rerank HTTP 调用失败。") from exc
        except ValueError as exc:
            raise KairoProviderError(f"{self.provider_name} rerank 返回的 JSON 不合法。") from exc
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError(f"{self.provider_name} rerank 调用失败：{type(exc).__name__}。") from exc
        return _results_from_index_payload(
            data,
            candidates,
            top_k=top_k,
            provider_name=self.provider_name,
            result_fields=self.result_fields,
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        client = self._client
        if client is not None:
            response = client.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as new_client:
                response = new_client.post(self.endpoint, headers=headers, json=payload)
        status_code = int(getattr(response, "status_code", 200) or 200)
        if status_code >= 400:
            raise KairoProviderError(f"{self.provider_name} rerank HTTP 状态异常：{status_code}。")
        return response


class CohereRerankProvider(_HTTPRerankProvider):
    """Cohere /v2/rerank 真实 provider。"""

    provider_name = "Cohere"
    endpoint = "https://api.cohere.com/v2/rerank"
    top_k_field = "top_n"
    result_fields = ("results",)

    def __init__(self, settings: KairoCloudSettings, client: httpx.Client | None = None) -> None:
        super().__init__(
            api_key=_secret_value(settings.cohere_api_key, "COHERE_API_KEY"),
            model=settings.cohere_rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            client=client,
        )


class JinaRerankProvider(_HTTPRerankProvider):
    """Jina /v1/rerank 真实 provider。"""

    provider_name = "Jina"
    endpoint = "https://api.jina.ai/v1/rerank"
    top_k_field = "top_n"
    result_fields = ("results",)

    def __init__(self, settings: KairoCloudSettings, client: httpx.Client | None = None) -> None:
        super().__init__(
            api_key=_secret_value(settings.jina_api_key, "JINA_API_KEY"),
            model=settings.jina_rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            client=client,
        )


class VoyageRerankProvider(_HTTPRerankProvider):
    """Voyage /v1/rerank 真实 provider。"""

    provider_name = "Voyage"
    endpoint = "https://api.voyageai.com/v1/rerank"
    top_k_field = "top_k"
    result_fields = ("data", "results")

    def __init__(self, settings: KairoCloudSettings, client: httpx.Client | None = None) -> None:
        super().__init__(
            api_key=_secret_value(settings.voyage_api_key, "VOYAGE_API_KEY"),
            model=settings.voyage_rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            client=client,
        )


class OpenAIListwiseRerankProvider:
    """使用 OpenAI LLM 做 listwise reranking 的 provider。"""

    def __init__(self, settings: KairoCloudSettings, llm_provider: LLMProvider | None = None) -> None:
        self.settings = settings
        if llm_provider is None:
            llm_provider = OpenAILLMProvider(settings)
            setattr(llm_provider, "model", settings.openai_rerank_model)
        self.llm_provider = llm_provider
        self.model = settings.openai_rerank_model

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        _validate_rerank_input(query, top_k)
        if not candidates:
            return []
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "你是 KairoRAG 的 listwise reranker。只根据候选文本与问题相关性排序。"
                    "只输出 JSON object，不输出 chain-of-thought。reason 只能是简短中文理由。"
                ),
            ),
            LLMMessage(role="user", content=_openai_listwise_prompt(query, candidates, top_k)),
        ]
        try:
            response = self.llm_provider.complete(
                messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = _parse_json_object(response.content)
        except Exception as exc:
            if isinstance(exc, KairoProviderError):
                raise
            raise KairoProviderError("OpenAI listwise rerank 返回内容不是合法 JSON。") from exc
        return _results_from_chunk_id_payload(payload, candidates, top_k=top_k, provider_name="OpenAI listwise")


class CrossEncoderRerankerProvider:
    """可选本地 cross-encoder reranker。"""

    def __init__(self, settings: KairoCloudSettings, model: Any | None = None) -> None:
        if not settings.cross_encoder_model:
            raise KairoProviderError("无法启用 cross_encoder reranker：缺少 CROSS_ENCODER_MODEL。")
        self.model_name = settings.cross_encoder_model
        self.model = model if model is not None else self._load_model(self.model_name)

    def rerank(self, query: str, candidates: list[RerankCandidate], *, top_k: int) -> list[RerankResult]:
        _validate_rerank_input(query, top_k)
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            raw_scores = self.model.predict(pairs)
        except Exception as exc:
            raise KairoProviderError("cross_encoder rerank 调用失败。") from exc
        scored: list[tuple[RerankCandidate, float]] = []
        for candidate, raw_score in zip(candidates, list(raw_scores)):
            scored.append((candidate, float(raw_score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RerankResult(chunk_id=candidate.chunk_id, score=score, metadata=dict(candidate.metadata))
            for candidate, score in scored[:top_k]
        ]

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - 可选依赖分支
            raise KairoProviderError(
                "无法启用 cross_encoder reranker：缺少 sentence-transformers 依赖；"
                "请安装可选依赖或改用 cloud reranker。"
            ) from exc
        return CrossEncoder(model_name)


def _validate_rerank_input(query: str, top_k: int) -> None:
    if not query or not query.strip():
        raise ValueError("query 不能为空。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")


def _secret_value(secret: Any, env_name: str) -> str:
    if secret is None or not secret.get_secret_value().strip():
        raise KairoProviderError(f"无法初始化 reranker：缺少 {env_name}。")
    return secret.get_secret_value()


def _results_from_index_payload(
    payload: dict[str, Any],
    candidates: list[RerankCandidate],
    *,
    top_k: int,
    provider_name: str,
    result_fields: tuple[str, ...],
) -> list[RerankResult]:
    raw_results: Any = []
    for field_name in result_fields:
        if isinstance(payload, dict) and isinstance(payload.get(field_name), list):
            raw_results = payload[field_name]
            break
    if not isinstance(raw_results, list):
        raise KairoProviderError(f"{provider_name} rerank 返回结果缺少 results/data 列表。")
    output: list[RerankResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            logger.warning("%s rerank 返回未知 index，已忽略。", provider_name)
            continue
        if index < 0 or index >= len(candidates):
            logger.warning("%s rerank 返回越界 index=%s，已忽略。", provider_name, index)
            continue
        candidate = candidates[index]
        score = _score_value(item.get("relevance_score", item.get("score")), provider_name)
        output.append(RerankResult(candidate.chunk_id, score, dict(candidate.metadata)))
    return output[:top_k]


def _results_from_chunk_id_payload(
    payload: dict[str, Any],
    candidates: list[RerankCandidate],
    *,
    top_k: int,
    provider_name: str,
) -> list[RerankResult]:
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise KairoProviderError(f"{provider_name} rerank 返回结果缺少 results 列表。")
    by_id = {candidate.chunk_id: candidate for candidate in candidates}
    output: list[RerankResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        candidate = by_id.get(chunk_id)
        if candidate is None:
            logger.warning("%s rerank 返回未知 chunk_id=%s，已忽略。", provider_name, chunk_id)
            continue
        score = _score_value(item.get("score"), provider_name)
        if score < 0 or score > 1:
            raise KairoProviderError(f"{provider_name} rerank score 必须在 0 到 1 之间。")
        metadata = dict(candidate.metadata)
        reason = str(item.get("reason") or "").strip()
        if reason:
            metadata["rerank_reason"] = reason[:200]
        output.append(RerankResult(chunk_id=chunk_id, score=score, metadata=metadata))
    output.sort(key=lambda item: item.score, reverse=True)
    return output[:top_k]


def _score_value(value: Any, provider_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise KairoProviderError(f"{provider_name} rerank 返回 score 不合法。") from exc


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON 输出必须是 object。")
    return payload


def _openai_listwise_prompt(query: str, candidates: list[RerankCandidate], top_k: int) -> str:
    items = []
    for candidate in candidates:
        title = candidate.metadata.get("title") or candidate.metadata.get("job_title") or ""
        preview = " ".join(candidate.text.split())[:700]
        items.append(
            {
                "chunk_id": candidate.chunk_id,
                "title": str(title),
                "text_preview": preview,
            }
        )
    return (
        "问题：\n"
        f"{query}\n\n"
        "候选 chunks：\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"请返回最相关的前 {top_k} 个候选。输出 JSON："
        '{"results":[{"chunk_id":"string","score":0.0,"reason":"简短中文理由"}]}'
    )
