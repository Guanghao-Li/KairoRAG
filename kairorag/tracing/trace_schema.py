"""Trace helpers for retrieval and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kairorag.schemas import utc_now


@dataclass
class TraceEvent:
    step: int
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "timestamp": self.timestamp,
        }

