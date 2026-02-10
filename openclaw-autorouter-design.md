# Building an autorouter for OpenClaw: architecture, approaches, and implementation

**The most practical path to intelligent LLM routing in OpenClaw is a LiteLLM-based proxy that combines three-axis classification (privacy × complexity × latency tolerance) with Presidio-powered PII detection, sitting between OpenClaw and your Ollama/Gemini backends.** This approach requires zero modifications to OpenClaw's codebase, works with its existing OpenAI-compatible provider interface, and can be incrementally upgraded from simple heuristics to learned routing. OpenClaw's plugin architecture also supports a native provider extension route, and a community plugin (ClawRouter) already demonstrates 14-dimension weighted scoring — but the proxy approach offers broader utility across all your tools, not just OpenClaw.

---

## OpenClaw's model system and where routing fits

OpenClaw is a **TypeScript monorepo** (~172K lines of TypeScript, ~550K total) built on Node.js ≥22 with pnpm workspaces. Its core agent toolkit is Pi (by Mario Zechner) — a TypeScript monorepo providing unified LLM API abstraction, agent runtime, terminal UI, and coding agent CLI. Pi routes requests to external providers; it does not perform inference itself. Pi is wrapped in a Gateway control plane that communicates over WebSocket on port 18789. The Gateway receives messages from 12+ messaging platforms (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, etc.) and dispatches them to Pi agent sessions for LLM inference.

Model selection today is straightforward and config-driven. The file `~/.openclaw/openclaw.json` defines a primary model plus ordered fallbacks under `agents.defaults.model`. When the primary model fails (rate limits, auth issues), OpenClaw rotates auth profiles within the provider, then falls back to the next model in the chain. Users can switch models mid-session with the **`/model` command** — typing `/model list` shows a numbered picker, `/model 3` selects by index, and `/model ollama/llama3.3` switches directly. There is no intelligence in this routing — it's purely manual selection with mechanical failover.

OpenClaw interfaces with Ollama through its **OpenAI-compatible API** (e.g., `http://localhost:11434/v1` on the LAN, or `http://127.0.0.1:11434/v1` if local). When `OLLAMA_API_KEY` is set, OpenClaw auto-discovers local models, queries their capabilities, and registers them with zero cost. It supports all major cloud providers natively (Anthropic, OpenAI, Google/Gemini, OpenRouter, Bedrock) and any OpenAI-compatible or Anthropic-compatible endpoint through custom `models.providers` configuration.

The critical insight for autorouter integration is OpenClaw's **four plugin slot types**: channel, tool, memory, and provider extensions. A provider extension can intercept model requests and reroute them — which is exactly what the community **ClawRouter plugin** (`@blockrun/clawrouter`) does. ClawRouter runs 14-dimension weighted scoring in <1ms, maps queries to four tiers (SIMPLE, MEDIUM, COMPLEX, REASONING), and routes to the cheapest capable model per tier. Installing it is as simple as `openclaw plugin install @blockrun/clawrouter` then `openclaw config set model blockrun/auto`. However, ClawRouter is tied to cloud models via x402 micropayments and doesn't handle local Ollama routing or privacy classification.

---

## The landscape of LLM routing: what works today

The field has matured rapidly since 2023, with two fundamental paradigms emerging. **Pre-generation routing** analyzes the query before any model runs and picks the best model upfront — this is what RouteLLM, NotDiamond, Martian, and most production systems use. **Post-generation routing** (cascade) sends queries to the cheapest model first, evaluates the output, and escalates if quality is insufficient — FrugalGPT and AutoMix take this approach. Pre-generation is faster; cascading is higher quality but burns extra tokens.

**RouteLLM** (LMSYS, ICLR 2025) is the most relevant open-source project. It routes between exactly two models — strong and weak — using four classifier strategies. The **matrix factorization (MF) router** performs best: trained on Chatbot Arena preference data. Without data augmentation, it achieves **95% of GPT-4's quality while routing 26% of queries to GPT-4** (~48% cost savings). With LLM judge augmentation (a meaningful additional training step), this drops to 14% GPT-4 calls (~75% cost savings). The routers transfer well across model pairs without retraining — a router trained on GPT-4/Mixtral works for Claude/Llama pairs. RouteLLM exposes an OpenAI-compatible server that uses LiteLLM as its backend, making it directly usable with Ollama endpoints. However, development stalled in August 2024 and it's limited to binary routing.

**NotDiamond** powers OpenRouter's `openrouter/auto` model with a trained meta-model that predicts optimal model selection per query in ~300ms. Their approach supports multi-model routing with Pareto optimization across quality, cost, and latency. The Python SDK is open-source, but the routing model itself is proprietary. **Martian** takes a unique approach using mechanistic interpretability — they claim to understand model internals well enough to predict performance without running the model, achieving up to 98% cost reduction. **Unify.ai** uses a neural network trained on continuously updated live benchmarks, with configurable quality/cost/speed sliders.

For a self-hosted setup, the most practical open-source options beyond RouteLLM are:

- **LiteLLM** — the most mature proxy with built-in routing strategies, Ollama/Gemini support, and semantic routing (BETA)
- **vLLM Semantic Router** (Red Hat) — enterprise-grade Rust+Go system with ModernBERT classification, 6 signal types including PII and complexity, but heavy infrastructure
- **Aurelio Semantic Router** — lightweight Python library using embeddings to match prompts to routes in ~10ms
- **Linx** — lightweight proxy specifically designed for local+cloud hybrid, with Ollama provider priority and automatic fallback

---

## How to estimate query complexity for routing decisions

The complexity estimation problem has multiple proven approaches, each trading off accuracy against overhead. For a local setup where router latency matters, the right strategy depends on your tolerance for classification errors versus response time.

**Rule-based heuristics** run in <1ms and catch obvious patterns. Token count signals complexity (short queries are usually simple); regex patterns detect code blocks, math expressions, and reasoning keywords ("prove," "analyze," "compare and contrast"). NVIDIA's LLM Router Blueprint v2 maps explicit intent descriptions to models — "chit chat" routes to a 9B model while "hard questions requiring deep reasoning" route to GPT-5. ClawRouter's 14-dimension scoring is essentially a sophisticated version of this: it weights keyword categories, applies a sigmoid for confidence calibration, then maps to tiers. These heuristics work surprisingly well for **70-80% of queries** where the complexity is unambiguous.

**Embedding-based semantic routing** offers a middle ground at ~10ms per decision. The Aurelio Semantic Router computes cosine similarity between the query embedding and embeddings of example utterances you define per route. You'd create routes like "simple_qa" (with examples: "What's the capital of France?", "Define photosynthesis"), "code_generation" (with examples: "Write a Python function to sort...", "Debug this React component"), and "complex_reasoning" (with examples: "Analyze the trade-offs between..."). This approach handles paraphrases and novel phrasings well without explicit keywords.

**Learned classifiers** deliver the highest accuracy. RouteLLM's MF router achieves the best quality-cost tradeoff among open-source options, trained on human preference data from Chatbot Arena. A fine-tuned BERT classifier can reach **~92% accuracy** in task classification with <20ms overhead. The most sophisticated approach — used by PRISM and HybridLLM — trains a model to predict the **quality gap** between your strong and weak models for each query, routing to the expensive model only when the gap is significant.

For your setup, the recommended progression is: start with rule-based heuristics (immediate value, zero overhead), add embedding-based routing when you identify patterns the heuristics miss, then optionally train a classifier once you've accumulated routing logs showing which model actually performed better.

---

## The proxy approach: LiteLLM as your routing layer

Building the autorouter as an **external proxy** rather than modifying OpenClaw directly is the stronger architectural choice for three reasons: it works with every tool that speaks OpenAI API (not just OpenClaw), it requires no upstream changes, and it separates routing concerns from application logic. LiteLLM is the best foundation — it's a mature FastAPI/Uvicorn proxy with **~8ms proxy overhead per request** (note: actual end-to-end P95 latency under load is 150-630ms depending on instance count; the 8ms figure represents proxy overhead only, measured against a fake backend), native Ollama support (`ollama_chat/model-name`), native Gemini support (`gemini/gemini-2.5-flash`), and 7 built-in routing strategies.

The core architecture evaluates three orthogonal axes per request — privacy, complexity, and latency tolerance — then selects the best model from the constrained candidate pool:

```
OpenClaw → http://localhost:4000/v1 (Autorouter Proxy)
                    │
    ┌── Axis 1: Privacy (Presidio) ──────────────┐
    │   → allowed_backends = local_only | any     │
    ├── Axis 2: Complexity (heuristics/embeddings)│
    │   → required_tier = simple..frontier        │
    ├── Axis 3: Latency (model name convention)   │
    │   → mode = interactive | background         │
    └─────────────────┬───────────────────────────┘
                      │
              Select best model FROM
              allowed_backends AT required_tier
              RESPECTING latency_mode
```

A practical LiteLLM configuration for your hardware:

```yaml
model_list:
  # Fast tier: Qwen3 8B — tool-capable, lightweight, GPU-resident
  - model_name: fast
    litellm_params:
      model: ollama_chat/qwen3-32k
      api_base: http://localhost:11434

  # Standard tier: Cogito 14B — tool-capable, hybrid reasoning, 128K context
  - model_name: standard
    litellm_params:
      model: ollama_chat/cogito:14b
      api_base: http://localhost:11434

  # Complex tier: Cogito 32B — tool-capable, best local agentic model
  - model_name: complex
    litellm_params:
      model: ollama_chat/cogito:32b
      api_base: http://localhost:11434

  # Coding tier: Qwen3-Coder-Next 80B/3B MoE — SWE-bench 70.6, 256K context
  - model_name: coding
    litellm_params:
      model: ollama_chat/qwen3-coder-next
      api_base: http://localhost:11434

  # Coding fallback: Devstral 24B — lighter, avoids cold-loading 51GB
  - model_name: coding-light
    litellm_params:
      model: ollama_chat/devstral-small-128k
      api_base: http://localhost:11434

  # Reasoning tier: Magistral 24B — tool-capable, reasoning-focused
  - model_name: reasoning
    litellm_params:
      model: ollama_chat/magistral
      api_base: http://localhost:11434

  # Vision tier: Qwen3-VL 8B — tool-capable, vision + tools
  - model_name: vision
    litellm_params:
      model: ollama_chat/qwen3-vl:8b-instruct-q4_K_M
      api_base: http://localhost:11434

  # Cloud tiers (deferred — local-only for Phase 1)
  # - model_name: cloud-fast
  #   litellm_params:
  #     model: gemini/gemini-2.5-flash
  #     api_key: os.environ/GEMINI_API_KEY
  # - model_name: cloud-frontier
  #   litellm_params:
  #     model: gemini/gemini-2.5-pro
  #     api_key: os.environ/GEMINI_API_KEY

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 2
  timeout: 120

# Note: LiteLLM's built-in Presidio guardrail requires deploying separate
# Presidio Analyzer and Anonymizer Docker containers and has active bugs
# (#12898, #8359). Presidio integration is deferred to the cloud tier phase.
# For Phase 1, all models are local — privacy axis is moot.
```

LiteLLM's custom routing can be extended by implementing a `CustomLogger` class that intercepts requests pre-call and modifies the target model. For more sophisticated routing, you can build a thin FastAPI layer in front of LiteLLM that classifies each request, selects the appropriate tier, and forwards it with the correct model name. This is also where you'd integrate RouteLLM's MF classifier or a semantic router.

To connect OpenClaw to this proxy, configure a custom provider:

```json
{
  "models": {
    "providers": {
      "autorouter": {
        "baseUrl": "http://127.0.0.1:4000/v1",
        "apiKey": "sk-local-proxy",
        "api": "openai-completions",
        "models": [{
          "id": "auto",
          "name": "Auto-routed (interactive)",
          "contextWindow": 32768,
          "cost": { "input": 0, "output": 0 }
        }, {
          "id": "auto-bg",
          "name": "Auto-routed (background)",
          "contextWindow": 32768,
          "cost": { "input": 0, "output": 0 }
        }]
      }
    }
  }
}
```

Then set `openclaw config set model autorouter/auto` and all queries flow through your routing layer.

---

## Three-axis routing: privacy × complexity × latency tolerance

The autorouter operates across three orthogonal dimensions that are evaluated independently and then combined. Privacy does **not** short-circuit complexity routing — it constrains which backends are eligible. Complexity determines the required model capability tier. Latency tolerance (foreground vs. background) determines whether to prefer already-loaded fast models or wait for the highest-quality option.

### Axis 1: Privacy classification with Presidio

The privacy gate uses **Microsoft Presidio**, an open-source PII detection framework combining regex matching, NER (spaCy-based), and context-aware analysis. It detects ~47 entity types (11 global + 36 country-specific) — names, SSNs, credit cards, medical IDs, bank numbers, passport numbers — with confidence scores per detection. Custom recognizers can extend this set. Detection runs in <10ms per 1,000-token request (the NLP pipeline includes spaCy tokenization, NER, regex, and context enhancement).

LiteLLM already integrates Presidio as a guardrail in its proxy configuration. You can configure entity-specific actions: **BLOCK** requests containing SSNs or credit cards entirely (forcing local routing), **MASK** names and emails before forwarding to cloud APIs, or allow clean queries through to any model.

A two-tier privacy classification: **Sensitive** (SSNs, credit cards, medical records, bank numbers, passports, personal names, email addresses, phone numbers) → constrain candidate pool to local Ollama models only. **Non-sensitive** (no PII detected above threshold) → all backends eligible, local and cloud.

The academic state-of-the-art is **PRISM** (November 2025), which implements a cloud-edge collaborative framework with entity-level differential privacy. For moderately sensitive prompts, PRISM sends a differentially private "sketch" to the cloud model, gets a structural response, then refines it locally with the actual sensitive values. This achieves only **1.54× latency overhead** compared to cloud-only inference while providing formal privacy guarantees. While PRISM is research-grade, its three-mode architecture (cloud-only, sketch-based collaboration, edge-only) is the right mental model for future evolution of the routing rules.

### Axis 2: Complexity classification

See the dedicated section above. The output is a capability tier: simple, standard, complex, or frontier.

### Axis 3: Foreground vs. background (latency tolerance)

The proxy is inherently blind to task context — it receives a standard OpenAI-compatible chat completion request with no metadata about the caller's context. The most practical solution is a **model name convention** that requires zero code changes to OpenClaw and zero protocol extensions:

```json
// In OpenClaw's openclaw.json — interactive sessions
{ "model": "autorouter/auto" }

// In cron/background agent configs
{ "model": "autorouter/auto-bg" }

// Optional: sub-second tool-call mode
{ "model": "autorouter/auto-urgent" }
```

The proxy parses the suffix to determine latency tolerance. This works because OpenClaw already supports per-agent model configuration, so background task agents can be pointed at a different model name without any codebase changes. Adding `auto-urgent` for sub-second agent tool calls requires just a new suffix and routing rule, no protocol negotiation.

Alternative identification strategies (usable as supplementary signals): streaming requests (`"stream": true`) are almost always interactive; non-streaming with large system prompts and structured output schemas tend to be batch; requests arriving in rapid bursts or at unusual hours suggest cron jobs. An in-codebase provider extension could propagate task metadata automatically if OpenClaw's task scheduler distinguishes foreground from background internally.

### The combined routing matrix

The three axes produce a constrained selection space. For each request, the router evaluates all three, then picks the best model from the intersection:

| Complexity | local_only + interactive | local_only + background | any + interactive | any + background |
|---|---|---|---|---|
| **simple** | Qwen3 8B (GPU) | Qwen3 8B (GPU) | Qwen3 8B (GPU) | Qwen3 8B (GPU) |
| **standard** | Cogito 14B (GPU) ³ | Cogito 14B (GPU) ³ | Gemini 2.5 Flash | Gemini 2.5 Flash |
| **complex** | 14B if loaded, else 32B ¹ | Cogito 32B (GPU+CPU) | Gemini 2.5 Flash | Gemini 2.5 Flash |
| **frontier** | 32B (best available local) ² | Qwen3 32B (GPU+CPU) | Gemini 2.5 Pro | Gemini 2.5 Pro |

¹ Interactive mode checks Ollama `/api/ps` — if 32B isn't loaded and 14B is GPU-resident, prefers 14B to avoid cold-load penalty (15-30s).
² Qwen3 dense models top out at 32B (the 235B-A22B MoE variant is a different architecture). In interactive + local_only + frontier, the router accepts the best currently-loaded model rather than forcing a cold load.
³ Phi-4 14B (originally specified) lacks native tool/function calling support in Ollama, which OpenClaw requires. Cogito 14B is the replacement: tool-capable, hybrid reasoning, 128K context.

Key behaviors encoded in the matrix: the cloud column still prefers local for simple tasks (no reason to burn API credits on trivia) but escalates to cloud earlier because it can. The local_only + interactive quadrant is the hardest — forced into local models but can't tolerate long load times, so the router strongly prefers whatever is already hot in VRAM. Background mode is willing to cold-load large models and wait for CPU-offloaded inference, always picking the highest-quality option.

### Implementation

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
SENSITIVE_ENTITIES = {
    "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "MEDICAL_LICENSE",
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE"
}

def classify_privacy(text: str) -> str:
    results = analyzer.analyze(text=text, language='en')
    if any(r.entity_type in SENSITIVE_ENTITIES and r.score > 0.6 for r in results):
        return "local_only"
    return "any"

def route(request):
    privacy = classify_privacy(extract_text(request.messages))
    complexity = classify_complexity(request.messages)

    # Axis 3: extract task type from model name convention
    model_requested = request.model  # "auto", "auto-bg", "auto-urgent"
    if model_requested.endswith("-bg"):
        latency_mode = "background"
    elif model_requested.endswith("-urgent"):
        latency_mode = "urgent"
    else:
        latency_mode = "interactive"

    # Get candidates for this complexity tier
    candidates = MODEL_TIERS[complexity]  # ordered list per tier

    # Axis 1: constrain by privacy
    if privacy == "local_only":
        candidates = [m for m in candidates if m.is_local]

    # Axis 3: apply latency preference
    if latency_mode in ("interactive", "urgent"):
        loaded_models = query_ollama_ps()  # GET /api/ps
        candidates = prioritize_loaded(candidates, loaded_models)
    else:
        # Background: prioritize quality, ignore load time
        candidates = prioritize_quality(candidates)

    return candidates[0]
```

---

## Hardware-aware routing for your specific setup

Your AMD RX 7800 XT with **16GB VRAM** and **128GB system RAM** creates a distinctive routing topology. Understanding what fits where is critical for the router's model selection logic.

At Q4_K_M quantization with 8K context, models up to **~14B parameters fit in VRAM** (~12-12.5GB, leaving ~3.5GB headroom on 16GB), 20B models fit tightly, and anything above 27B requires CPU offloading. Two 8B models can coexist in VRAM simultaneously, or one 14B plus one 3-4B model. The KV cache is the hidden VRAM consumer — an 8B model's KV cache grows from ~0.3GB at 2K context to ~5GB at 32K context, so long-context queries on larger models can push past VRAM limits unexpectedly.

Your **128GB RAM is a major advantage** — it enables running 32B and even 70B models with partial GPU offload. A 32B Q4_K_M model (~20GB) can load ~60% of layers to GPU and the rest to RAM, delivering **5-15 tok/s**. A 70B model (~42.5GB) runs mostly from RAM at **2-5 tok/s**. This is slow but usable for complex queries where quality matters more than speed.

Critical Ollama settings for routing performance:

- **`OLLAMA_KEEP_ALIVE=-1`** for your primary model (prevents unloading)
- **`OLLAMA_MAX_LOADED_MODELS=2`** to allow a fast model + quality model simultaneously
- **`OLLAMA_FLASH_ATTENTION=1`** for better AMD/ROCm performance
- Avoid frequent model switching — cold-loading a 14B model takes 15-30 seconds from NVMe SSD (community reports 24-29s for ~9GB model files), and loading triggers unloading of the current GPU-resident model

The router should be aware of these constraints. A practical tier mapping for your hardware:

| Tier | Model | Location | Speed | Use case |
|------|-------|----------|-------|----------|
| Fast | Qwen3 8B (qwen3-32k) | GPU (always loaded) | 35-45 tok/s | Simple Q&A, classification, formatting |
| Standard | Cogito 14B (cogito:14b) | GPU (on demand) | 20-30 tok/s | General tasks, summarization, moderate reasoning |
| Complex | Cogito 32B (cogito:32b) | GPU+CPU split | 5-15 tok/s | Deep analysis, multi-step tasks |
| Coding | Qwen3-Coder-Next 80B/3B MoE | RAM-heavy | varies | Code generation, SWE-bench 70.6 |
| Coding (light) | Devstral 24B | GPU (on demand) | 15-25 tok/s | Coding when cold-load of 51GB unacceptable |
| Reasoning | Magistral 24B | GPU (on demand) | 15-25 tok/s | Math, proofs, formal logic |
| Vision | Qwen3-VL 8B | GPU (on demand) | 30-40 tok/s | Image understanding + tool calling |
| Cloud Fast | Gemini 2.5 Flash | API | ~80 tok/s | Non-sensitive complex tasks, long context |
| Cloud Frontier | Gemini 2.5 Pro | API | ~40 tok/s | Hardest reasoning, agent workflows |

Note: All local models listed above have native tool/function calling support, which OpenClaw requires for agentic workflows. Models without tool calling (Phi-4, DeepSeek-R1, Gemma3, Mixtral, etc.) are excluded from the tier mapping despite being available on the Ollama host.

The router should factor in **model loading state** — if the 14B model is already loaded, prefer it over the 8B model for borderline queries rather than paying the switching cost. Querying Ollama's `/api/ps` endpoint reveals which models are currently loaded.

---

## Putting it together: a staged implementation plan

**Phase 1 — Proxy with static tiers (1 day).** Install LiteLLM, configure Ollama models on the Ollama host in `config.yaml` with the tier structure above (cloud tiers deferred). Set up fallback chains within local tiers. Point OpenClaw to the autorouter as a custom provider. At this stage, use LiteLLM's `model_group_alias` to manually map model names to tiers and switch via OpenClaw's `/model` command. You get unified API, fallbacks, and logging immediately.

**Phase 2 — Rule-based auto-routing (2-3 days).** Build a Python middleware (FastAPI, ~400-600 lines) that sits in front of LiteLLM. Implement the three-axis classification: keyword-based complexity scoring (detect code blocks, math notation, multi-step instructions, reasoning markers), Presidio for privacy classification, and model name convention parsing for latency tolerance (`auto` vs. `auto-bg` vs. `auto-urgent`). The middleware receives requests at `/v1/chat/completions`, evaluates all three axes, selects the best model from the constrained candidate pool, and forwards to LiteLLM. Register `autorouter/auto` and `autorouter/auto-bg` in OpenClaw's provider config.

**Phase 3 — Semantic routing (1 week).** Replace or augment keyword heuristics with embedding-based classification using Aurelio's Semantic Router. Define route categories with 10-20 example utterances each. Use a local embedding model (e.g., `nomic-embed-text` via Ollama) to avoid cloud dependency for the router itself. This handles novel phrasings that keywords miss.

**Phase 4 — Learned routing (ongoing).** Log all routing decisions and user feedback. Once you have 500+ labeled examples, train a RouteLLM-style MF classifier on your actual usage patterns. This is where you'll see the biggest quality improvements — a router tuned to your specific mix of queries will outperform any generic approach.

The alternative **in-codebase approach** — building an OpenClaw provider extension — is cleaner for the OpenClaw ecosystem and could be contributed upstream. The extension would implement the provider plugin interface, intercept `createAgentSession` calls in `src/agents/pi-embedded-runner/model.ts`, and inject routing logic into the `ModelRegistry`. The TypeBox schema system provides type-safe configuration. However, this locks the router to OpenClaw only and requires TypeScript expertise — the proxy approach serves all your tools simultaneously.

## Conclusion

The LLM routing space has converged on proven approaches: **pre-generation classification** (heuristic or learned) for speed, **binary or tiered model selection** for simplicity, and **PII detection as a privacy gate** before any cloud call. For your specific setup — 20 Ollama models on 16GB VRAM + 128GB RAM + Gemini API — the binding constraints are VRAM capacity and model loading latency, not routing algorithm sophistication. A LiteLLM proxy with three-axis routing (privacy × complexity × latency tolerance) and Presidio will capture **70-85% of potential savings** with minimal complexity. The model name convention (`auto`, `auto-bg`, `auto-urgent`) provides foreground/background differentiation with zero code changes. The remaining gains from learned routing are real but incremental. Start with the proxy, accumulate data, and let your routing evolve from heuristics to embeddings to trained classifiers as your usage patterns become clear. The key architectural decision — proxy versus in-codebase — favors the proxy: it's tool-agnostic, language-agnostic, and the existing ClawRouter plugin proves that OpenClaw's provider system will seamlessly consume whatever your proxy serves.