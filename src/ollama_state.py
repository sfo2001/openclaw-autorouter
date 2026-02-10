"""Ollama state polling with TTL cache.

Queries the Ollama /api/ps endpoint to determine which models are currently
loaded in memory. Results are cached for 5 seconds to avoid excessive polling.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CACHE_TTL_SECONDS = 5.0
_REQUEST_TIMEOUT = 3.0

_cache_models: list[str] = []
_cache_timestamp: float = 0.0


async def get_loaded_models() -> list[str]:
    """Return list of currently loaded Ollama model names.

    Cached for CACHE_TTL_SECONDS. Returns empty list on error.
    """
    global _cache_models, _cache_timestamp

    now = time.monotonic()
    if now - _cache_timestamp < CACHE_TTL_SECONDS:
        return _cache_models

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            resp.raise_for_status()
            data = resp.json()

        models = []
        for entry in data.get("models", []):
            name = entry.get("name", "")
            if name:
                models.append(name)

        _cache_models = models
        _cache_timestamp = now
        logger.debug("Ollama loaded models: %s", models)
        return models

    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("Failed to query Ollama /api/ps: %s", exc)
        # On error, invalidate cache and return empty (assume nothing loaded)
        _cache_models = []
        _cache_timestamp = 0.0
        return []


def invalidate_cache() -> None:
    """Force cache refresh on next call. Useful for testing."""
    global _cache_models, _cache_timestamp
    _cache_models = []
    _cache_timestamp = 0.0
