"""基础生产观测能力：结构化 trace、JSONL 写入和敏感信息脱敏。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECRET_KEY_MARKERS = ("api_key", "authorization", "token", "secret", "password")
SECRET_VALUE_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    timestamp: str
    run_id: str
    query: str | None
    detail: dict[str, Any]


class TraceLogger:
    """把结构化事件追加写入 JSONL 文件。"""

    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def log(self, event: TraceEvent) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = redact_secrets(asdict(event))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(safe_json_dumps(payload) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dumps(data: Any) -> str:
    return json.dumps(redact_secrets(data), ensure_ascii=False, default=_json_default, sort_keys=True)


def redact_secrets(data: Any) -> Any:
    if is_dataclass(data):
        return redact_secrets(asdict(data))
    if isinstance(data, dict):
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in SECRET_KEY_MARKERS):
                redacted[key] = "[已隐藏]"
            else:
                redacted[key] = redact_secrets(value)
        return redacted
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_secrets(item) for item in data)
    if isinstance(data, str):
        return SECRET_VALUE_PATTERN.sub("[已隐藏]", data)
    return data


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)
