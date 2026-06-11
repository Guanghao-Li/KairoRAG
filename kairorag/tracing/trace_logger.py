"""Small in-memory trace logger."""

from __future__ import annotations

from typing import Any

from kairorag.tracing.trace_schema import TraceEvent


class TraceLogger:
    """Collect ordered trace events for a single run."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def add(self, tool: str, inputs: dict[str, Any] | None = None, outputs: dict[str, Any] | None = None) -> None:
        self.events.append(
            TraceEvent(
                step=len(self.events) + 1,
                tool=tool,
                inputs=inputs or {},
                outputs=outputs or {},
            )
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

