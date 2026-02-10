"""FastAPI auto-routing middleware for OpenClaw Autorouter.

Sits in front of LiteLLM proxy. Routes `auto`, `auto-bg`, and `auto-urgent`
model requests through classification + model selection. All other model names
pass through unchanged to LiteLLM.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.classifier import classify_request
from src.ollama_state import get_loaded_models
from src.router import select_model
from src.stats import get_stats, record_auto_route, record_passthrough

logger = logging.getLogger(__name__)

LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4001")
AUTO_MODEL_PREFIXES = ("auto", "auto-bg", "auto-urgent")

# 50 MB: accommodates base64-encoded images for vision models (5-10 MB each)
MAX_BODY_BYTES = 50 * 1024 * 1024

# Models exposed as auto-routing endpoints
AUTO_MODELS = [
    {"id": "auto", "object": "model", "owned_by": "autorouter"},
    {"id": "auto-bg", "object": "model", "owned_by": "autorouter"},
    {"id": "auto-urgent", "object": "model", "owned_by": "autorouter"},
]

app = FastAPI(title="OpenClaw Autorouter", version="2.0.0")

# Shared httpx client (created at startup, closed at shutdown)
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _http_client
    _http_client = httpx.AsyncClient(
        base_url=LITELLM_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )
    logger.info("Autorouter started, LiteLLM backend: %s", LITELLM_BASE)


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


def _is_auto_model(model: str) -> bool:
    return model in AUTO_MODEL_PREFIXES


async def _proxy_streaming(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> StreamingResponse:
    """Stream a proxied response from LiteLLM."""
    assert _http_client is not None
    req = _http_client.build_request(method, path, headers=headers, content=body)
    resp = await _http_client.send(req, stream=True)

    async def _stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    # Forward relevant response headers
    response_headers: dict[str, str] = {}
    for key in ("content-type", "x-autorouter-model"):
        if key in resp.headers:
            response_headers[key] = resp.headers[key]

    return StreamingResponse(
        _stream(),
        status_code=resp.status_code,
        headers=response_headers,
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Main route: classify auto requests, proxy to LiteLLM."""
    body_bytes = await request.body()

    if len(body_bytes) > MAX_BODY_BYTES:
        return JSONResponse(
            content={
                "error": {
                    "message": "Request body too large",
                    "type": "invalid_request_error",
                }
            },
            status_code=413,
        )

    try:
        body: dict[str, Any] = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            content={
                "error": {
                    "message": "Invalid JSON in request body",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    model = body.get("model", "")

    # Forward headers (strip host, keep auth)
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    if _is_auto_model(model):
        # Classify and route
        messages = body.get("messages", [])
        decision = classify_request(messages, model)
        loaded_models = await get_loaded_models()
        result = select_model(
            decision.task_type,
            decision.complexity,
            decision.latency_mode,
            loaded_models,
        )

        hot_str = "hot" if result.was_hot else "cold"
        fallback_str = "fallback" if result.was_fallback else "ideal"

        if result.was_fallback:
            logger.info(
                "Auto-route: %s -> %s [task=%s complexity=%s latency=%s %s %s ideal=%s]",
                model,
                result.tier,
                decision.task_type.value,
                decision.complexity.value,
                decision.latency_mode.value,
                hot_str,
                fallback_str,
                result.ideal_tier,
            )
        else:
            logger.info(
                "Auto-route: %s -> %s [task=%s complexity=%s latency=%s %s %s]",
                model,
                result.tier,
                decision.task_type.value,
                decision.complexity.value,
                decision.latency_mode.value,
                hot_str,
                fallback_str,
            )

        record_auto_route(
            task_type=decision.task_type.value,
            complexity=decision.complexity.value,
            latency_mode=decision.latency_mode.value,
            tier=result.tier,
            was_hot=result.was_hot,
            was_fallback=result.was_fallback,
        )

        selected_tier = result.tier

        # Rewrite model in body
        body["model"] = selected_tier
        body_bytes = json.dumps(body).encode()
        forward_headers["content-type"] = "application/json"

        # Add routing header for observability
        forward_headers["x-autorouter-decision"] = (
            f"{decision.task_type.value}/{decision.complexity.value}"
        )
    else:
        selected_tier = model
        record_passthrough(model)

    is_streaming = body.get("stream", False)

    if is_streaming:
        resp = await _proxy_streaming("POST", "/v1/chat/completions", forward_headers, body_bytes)
        resp.headers["x-autorouter-model"] = selected_tier
        return resp

    # Non-streaming: proxy and return JSON
    assert _http_client is not None
    litellm_resp = await _http_client.post(
        "/v1/chat/completions",
        headers=forward_headers,
        content=body_bytes,
    )
    response = JSONResponse(
        content=litellm_resp.json(),
        status_code=litellm_resp.status_code,
    )
    response.headers["x-autorouter-model"] = selected_tier
    return response


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Proxy model list from LiteLLM and add auto-routing models."""
    assert _http_client is not None

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    try:
        resp = await _http_client.get("/v1/models", headers=forward_headers)
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        data = {"object": "list", "data": []}

    # Append auto models
    existing_ids = {m.get("id") for m in data.get("data", [])}
    for auto_model in AUTO_MODELS:
        if auto_model["id"] not in existing_ids:
            data.setdefault("data", []).append(auto_model)

    return JSONResponse(content=data)


@app.get("/stats")
async def stats() -> JSONResponse:
    """Return routing statistics since server start."""
    return JSONResponse(content=get_stats())


@app.get("/health")
async def health() -> JSONResponse:
    """Health check: verify LiteLLM backend is reachable."""
    assert _http_client is not None

    try:
        resp = await _http_client.get("/health/liveliness")
        litellm_ok = resp.status_code == 200
    except httpx.HTTPError:
        litellm_ok = False

    status = "healthy" if litellm_ok else "degraded"
    return JSONResponse(
        content={"status": status, "litellm": litellm_ok},
        status_code=200 if litellm_ok else 503,
    )
