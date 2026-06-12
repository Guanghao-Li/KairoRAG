"""BM25 keyword search provider。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from kairorag.providers.errors import KairoProviderError


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+#-][A-Za-z0-9]+)*|[\u4e00-\u9fff]")
SUPPORTED_FILTER_KEYS = {"archived", "verification_status", "source_type"}


@dataclass(frozen=True)
class KeywordDocument:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KeywordSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any]


class KeywordSearchProvider(Protocol):
    def index(self, documents: list[KeywordDocument]) -> None:
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[KeywordSearchResult]:
        ...


class BM25KeywordSearchProvider:
    """使用 rank_bm25.BM25Okapi 的真实 BM25 关键词检索。"""

    def __init__(self, title_weight: int = 3) -> None:
        if title_weight < 1:
            raise ValueError("title_weight 必须大于等于 1。")
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:  # pragma: no cover - 依赖缺失分支
            raise KairoProviderError("无法初始化 BM25 provider：缺少 rank-bm25 依赖。") from exc
        self._bm25_cls = BM25Okapi
        self.title_weight = title_weight
        self.documents: list[KeywordDocument] = []
        self._bm25: Any | None = None

    def index(self, documents: list[KeywordDocument]) -> None:
        self.documents = list(documents)
        if not self.documents:
            self._bm25 = None
            return
        corpus = [_tokenize(_weighted_text(document, self.title_weight)) for document in self.documents]
        self._bm25 = self._bm25_cls(corpus)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[KeywordSearchResult]:
        if top_k <= 0 or self._bm25 is None:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        results: list[KeywordSearchResult] = []
        for document, score in zip(self.documents, scores):
            if score <= 0 or not _metadata_matches(document.metadata, filters):
                continue
            results.append(
                KeywordSearchResult(
                    chunk_id=document.chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    score=float(score),
                    metadata=document.metadata,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _weighted_text(document: KeywordDocument, title_weight: int) -> str:
    return " ".join([document.title] * title_weight + [document.text])


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    filters = filters or {}
    for key, expected in filters.items():
        if key not in SUPPORTED_FILTER_KEYS:
            continue
        if metadata.get(key) != expected:
            return False
    return True
