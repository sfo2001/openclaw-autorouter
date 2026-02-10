"""In-memory routing statistics since server start.

Thread-safe for single-process async (Counter operations are atomic in CPython).
No external dependencies.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

_start_time: float = time.monotonic()

_total_requests: int = 0
_auto_requests: int = 0
_passthrough_requests: int = 0

_by_task_type: Counter[str] = Counter()
_by_complexity: Counter[str] = Counter()
_by_latency_mode: Counter[str] = Counter()
_by_selected_tier: Counter[str] = Counter()
_by_hot_cold: Counter[str] = Counter()
_by_fallback: Counter[str] = Counter()


def record_auto_route(
    task_type: str,
    complexity: str,
    latency_mode: str,
    tier: str,
    *,
    was_hot: bool,
    was_fallback: bool,
) -> None:
    """Record an auto-routed request."""
    global _total_requests, _auto_requests

    _total_requests += 1
    _auto_requests += 1
    _by_task_type[task_type] += 1
    _by_complexity[complexity] += 1
    _by_latency_mode[latency_mode] += 1
    _by_selected_tier[tier] += 1
    _by_hot_cold["hot" if was_hot else "cold"] += 1
    _by_fallback["fallback" if was_fallback else "ideal"] += 1


def record_passthrough(model: str) -> None:
    """Record a passthrough (non-auto) request."""
    global _total_requests, _passthrough_requests

    _total_requests += 1
    _passthrough_requests += 1
    _by_selected_tier[model] += 1


def get_stats() -> dict[str, Any]:
    """Return JSON-serializable snapshot of all counters."""
    return {
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "total_requests": _total_requests,
        "auto_requests": _auto_requests,
        "passthrough_requests": _passthrough_requests,
        "by_task_type": dict(_by_task_type),
        "by_complexity": dict(_by_complexity),
        "by_latency_mode": dict(_by_latency_mode),
        "by_selected_tier": dict(_by_selected_tier),
        "by_hot_cold": dict(_by_hot_cold),
        "by_fallback": dict(_by_fallback),
    }


def reset() -> None:
    """Reset all counters. Used in tests."""
    global _start_time, _total_requests, _auto_requests, _passthrough_requests

    _start_time = time.monotonic()
    _total_requests = 0
    _auto_requests = 0
    _passthrough_requests = 0
    _by_task_type.clear()
    _by_complexity.clear()
    _by_latency_mode.clear()
    _by_selected_tier.clear()
    _by_hot_cold.clear()
    _by_fallback.clear()
