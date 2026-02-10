# OpenClaw Autorouter

LLM routing proxy for [OpenClaw](https://github.com/nicepkg/openclaw). Routes requests to Ollama models based on task tier using LiteLLM as the proxy backend.

## Architecture

```
Client -> Autorouter (:4000) -> LiteLLM (:4001, internal) -> Ollama (localhost:11434)
```

Phase 2 adds auto-routing: requests to `auto`, `auto-bg`, and `auto-urgent` model names are classified by task type (coding/reasoning/vision/general) and complexity, then routed to the best model tier. Other model names (`fast`, `standard`, etc.) pass through to LiteLLM unchanged.

## Model Tiers

All models require native tool/function calling support (OpenClaw requirement).

| Tier | Model | Size | Use Case |
|------|-------|------|----------|
| `fast` | Qwen3 8B (qwen3-32k-fast) | 5.2GB | Simple Q&A, classification |
| `standard` | Cogito 14B | 9.0GB | General tasks, summarization |
| `complex` | Cogito 32B | 19GB | Deep analysis, multi-step tasks |
| `coding` | Qwen3-Coder-Next 80B/3B MoE | 51GB | Code generation (SWE-bench 70.6) |
| `coding-light` | Devstral 24B | 15GB | Coding without 51GB cold-load |
| `thinking` | Qwen3 8B (qwen3-32k) | 5.2GB | Chain-of-thought reasoning |
| `reasoning` | Cogito 32B | 19GB | Math, proofs, formal logic |
| `vision` | Qwen3-VL 8B | 6.1GB | Image understanding |

## Prerequisites

- Docker and docker-compose on the autorouter host
- Ollama running with the models listed above
- Ollama host reachable from the Docker container

## Quick Start

```bash
# 1. Copy config templates
cp .env.example .env
cp config.yaml.example config.yaml

# 2. Edit .env with your Ollama host and API key
#    Edit config.yaml to set api_base for each model tier

# 3. Start
docker-compose up
```

The autorouter listens on port 4000. Verify with:

```bash
curl http://localhost:4000/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint (used by routing middleware) |
| `LITELLM_MASTER_KEY` | `sk-change-me` | API key for LiteLLM proxy |
| `OLLAMA_HOST` | `localhost` | Ollama SSH host (used by `bench.py`) |
| `AUTOROUTER_URL` | `http://localhost:4000/v1/chat/completions` | Autorouter endpoint (used by `bench.py`) |
| `AUTOROUTER_API_KEY` | `sk-change-me` | API key for benchmark requests |

See `.env.example` for all variables including deployment options.

### LiteLLM Config

Edit `config.yaml` (copied from `config.yaml.example`) to configure model tiers. Each tier maps a model name to an Ollama model and endpoint. Set `api_base` to your Ollama host URL.

### Deployment

`build-and-push.sh` builds the Docker image, pushes to a registry, and deploys via SSH. Configure with env vars:

```bash
REGISTRY=your-registry:5000 DEPLOY_HOST=your-server ./build-and-push.sh
```

An Unraid Docker template is provided in `unraid-template.xml`.

## OpenClaw Integration

Add the autorouter as a custom provider in `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "autorouter": {
        "baseUrl": "http://localhost:4000/v1",
        "apiKey": "sk-change-me",
        "api": "openai-completions",
        "models": [
          { "id": "fast", "name": "Fast (Qwen3 8B)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } },
          { "id": "standard", "name": "Standard (Cogito 14B)", "contextWindow": 131072, "cost": { "input": 0, "output": 0 } },
          { "id": "complex", "name": "Complex (Cogito 32B)", "contextWindow": 131072, "cost": { "input": 0, "output": 0 } },
          { "id": "coding", "name": "Coding (Qwen3-Coder-Next)", "contextWindow": 262144, "cost": { "input": 0, "output": 0 } },
          { "id": "coding-light", "name": "Coding Light (Devstral 24B)", "contextWindow": 131072, "cost": { "input": 0, "output": 0 } },
          { "id": "thinking", "name": "Thinking (Qwen3 8B)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } },
          { "id": "reasoning", "name": "Reasoning (Cogito 32B)", "contextWindow": 131072, "cost": { "input": 0, "output": 0 } },
          { "id": "vision", "name": "Vision (Qwen3-VL 8B)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } },
          { "id": "auto", "name": "Auto (interactive)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } },
          { "id": "auto-bg", "name": "Auto (background)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } },
          { "id": "auto-urgent", "name": "Auto (urgent)", "contextWindow": 32768, "cost": { "input": 0, "output": 0 } }
        ]
      }
    }
  }
}
```

Select a tier: `/model autorouter/fast`, `/model autorouter/coding`, `/model autorouter/auto`, etc.

## Running Tests

```bash
python3 -m venv venv && venv/bin/pip install -e ".[dev]"
make check  # runs lint + typecheck + tests
```

## Phases

1. **Phase 1** (complete): Static proxy with manual tier selection
2. **Phase 2** (current): Rule-based auto-routing (`auto`, `auto-bg`, `auto-urgent`)
3. **Phase 3**: Embedding-based semantic routing
4. **Phase 4**: Learned routing with labeled data

See `openclaw-autorouter-design.md` for the full architecture.

## File Structure

```
Dockerfile              # LiteLLM + FastAPI dual-service image
docker-compose.yml      # Development (mounts src/ for live editing)
entrypoint.sh           # Starts LiteLLM on :4001, then FastAPI on :4000
config.yaml.example     # LiteLLM model tier config template
.env.example            # Environment variable template
src/
  main.py               # FastAPI app: /v1/chat/completions, /v1/models, /health
  classifier.py         # Task type + complexity detection (bilingual DE/EN)
  router.py             # Model selection matrix with interactive fallback
  ollama_state.py       # Ollama /api/ps polling with 5s TTL cache
tests/
  test_classifier.py    # Classification tests
  test_router.py        # Routing logic tests
bench.py                # Tier benchmark (cold/hot TTFT, tok/s, task eval)
build-and-push.sh       # Build + push + deploy via SSH
unraid-template.xml     # Unraid Docker GUI template
```
