#!/usr/bin/env bash
set -euo pipefail

# Phase 2 entrypoint: starts LiteLLM (port 4001) + FastAPI middleware (port 4000)

LITELLM_PORT="${LITELLM_PORT:-4001}"
FASTAPI_PORT="${FASTAPI_PORT:-4000}"

# Health check using Python (curl not available in LiteLLM base image)
check_health() {
    python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:${LITELLM_PORT}/health/liveliness', timeout=2)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
"
}

# Start LiteLLM proxy in background
echo "Starting LiteLLM on port ${LITELLM_PORT}..."
litellm --config /app/config.yaml --port "${LITELLM_PORT}" &
LITELLM_PID=$!

# Wait for LiteLLM to be ready
echo "Waiting for LiteLLM health..."
for i in $(seq 1 30); do
    if check_health; then
        echo "LiteLLM ready on port ${LITELLM_PORT}"
        break
    fi
    if ! kill -0 "$LITELLM_PID" 2>/dev/null; then
        echo "ERROR: LiteLLM process died"
        exit 1
    fi
    sleep 1
done

# Verify LiteLLM actually started
if ! check_health; then
    echo "ERROR: LiteLLM failed to start within 30 seconds"
    kill "$LITELLM_PID" 2>/dev/null || true
    exit 1
fi

# Trap signals to clean up both processes
cleanup() {
    echo "Shutting down..."
    kill "$LITELLM_PID" 2>/dev/null || true
    wait "$LITELLM_PID" 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT

# Start FastAPI middleware in foreground
echo "Starting autorouter middleware on port ${FASTAPI_PORT}..."
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${FASTAPI_PORT}" \
    --log-level info
