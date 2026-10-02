from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolOptions:
    """Dialt-specific behavior for a LiveKit function tool."""

    read_only: bool | None = None
    requires_permission: bool | None = None
    permission_source: Literal["caller", "application"] | None = None
    resolver_only: bool | None = None
    expected_duration: Literal["instant", "seconds", "long"] | None = None
    status_label: str | None = None
    timeout: float | None = None
    deferred: bool | None = None
    deferred_timeout: float | None = None
    notify_on_complete: bool | None = None
