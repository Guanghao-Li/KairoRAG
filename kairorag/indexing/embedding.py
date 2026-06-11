"""Local deterministic embedding provider."""

from __future__ import annotations

import hashlib
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_#+.-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize text for keyword and hashing embeddings."""

    return [token.lower() for token in TOKEN_RE.findall(text or "")]


class HashingEmbeddingProvider:
    """A tiny local embedding provider with no model downloads."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine for normalized or near-normalized vectors."""

    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))

