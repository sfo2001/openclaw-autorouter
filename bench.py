#!/usr/bin/env python3
"""Autorouter tier benchmark: cold load, hot call, TTFT, tok/s, task-specific eval."""

import json
import os
import subprocess
import time
import urllib.request
from typing import Any

AUTOROUTER_URL = os.environ.get("AUTOROUTER_URL", "http://localhost:4000/v1/chat/completions")
API_KEY = os.environ.get("AUTOROUTER_API_KEY", "sk-change-me")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
MAX_TOKENS = 1024

# Generic prompt for timing benchmarks
GENERIC_PROMPT = "Explain in detail how a hash table works, including collision resolution strategies. Be thorough but concise."

# Task-specific prompts matched to each tier's intended use case
TASK_PROMPTS: dict[str, dict[str, str | int]] = {
    "fast": {
        "prompt": "What are the three primary colors? List them.",
        "max_tokens": 128,
        "criteria": "Quick, correct, no fluff",
    },
    "standard": {
        "prompt": "Compare and contrast REST and GraphQL APIs. Cover strengths, weaknesses, and when to use each.",
        "max_tokens": 1024,
        "criteria": "Balanced, well-structured, accurate",
    },
    "complex": {
        "prompt": "Design a rate limiter for a distributed API gateway. Cover the algorithm choice (token bucket vs sliding window), how to handle distributed state across multiple nodes, and failure modes. Include pseudocode.",
        "max_tokens": 2048,
        "criteria": "Deep analysis, multi-step design, pseudocode quality",
    },
    "coding": {
        "prompt": "Write a Python function that implements a LRU cache with O(1) get and put operations. Include type hints, docstring, and unit tests using pytest.",
        "max_tokens": 2048,
        "criteria": "Correct implementation, clean code, working tests",
    },
    "coding-light": {
        "prompt": "Write a bash script that monitors disk usage and sends an alert if any partition exceeds 90%. Include error handling.",
        "max_tokens": 1024,
        "criteria": "Working script, proper error handling, clear output",
    },
    "thinking": {
        "prompt": "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How does the farmer get everything across safely? Show your step-by-step reasoning.",
        "max_tokens": 8192,
        "criteria": "Correct solution, visible chain-of-thought",
    },
    "reasoning": {
        "prompt": "Prove that the square root of 2 is irrational using proof by contradiction. Be rigorous and precise.",
        "max_tokens": 2048,
        "criteria": "Rigorous proof, correct logic, mathematical precision",
    },
    "vision": {
        "prompt": "You are a vision model. Describe how you would analyze an image of a circuit board to identify damaged components. What visual features would you look for?",
        "max_tokens": 1024,
        "criteria": "Technical depth, practical methodology",
    },
}

TIERS = [
    "fast",
    "standard",
    "complex",
    "coding",
    "coding-light",
    "thinking",
    "reasoning",
    "vision",
]


OLLAMA_IDLE_POLL_INTERVAL = 5  # seconds between idle checks
OLLAMA_IDLE_TIMEOUT = 300  # max seconds to wait for idle


def ollama_get_loaded_models():
    """Return list of model names currently loaded in Ollama."""
    result = subprocess.run(
        ["ssh", OLLAMA_HOST, "ollama ps"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    models = []
    for line in result.stdout.strip().split("\n")[1:]:
        name = line.split()[0] if line.strip() else None
        if name:
            models.append(name)
    return models


def ollama_wait_idle():
    """Wait until Ollama has no loaded models (fully idle).

    Polls ollama ps every OLLAMA_IDLE_POLL_INTERVAL seconds.
    Large models on GPU+CPU split can take minutes to finish inference
    and unload, even after ollama stop is sent.
    """
    deadline = time.monotonic() + OLLAMA_IDLE_TIMEOUT
    while time.monotonic() < deadline:
        models = ollama_get_loaded_models()
        if not models:
            return
        remaining = int(deadline - time.monotonic())
        print(f"    Waiting for Ollama to idle (loaded: {', '.join(models)}) [{remaining}s left]")
        time.sleep(OLLAMA_IDLE_POLL_INTERVAL)
    print(f"    WARNING: Ollama not idle after {OLLAMA_IDLE_TIMEOUT}s, proceeding anyway")


def ollama_stop_all():
    """Stop all loaded models and wait for Ollama to become fully idle."""
    models = ollama_get_loaded_models()
    for model_name in models:
        subprocess.run(
            ["ssh", OLLAMA_HOST, f"ollama stop {model_name}"],
            capture_output=True,
            timeout=15,
        )
    if models:
        ollama_wait_idle()


def ollama_ps_context():
    """Return the context size of the currently loaded model."""
    result = subprocess.run(
        ["ssh", OLLAMA_HOST, "ollama ps"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for line in result.stdout.strip().split("\n")[1:]:
        parts = line.split()
        for p in parts:
            if p.isdigit() and int(p) > 1000:
                return p
    return "?"


def stream_request(tier: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    """Send a streaming chat request and measure timing."""
    body = json.dumps(
        {
            "model": tier,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode()

    req = urllib.request.Request(
        AUTOROUTER_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    t_start = time.monotonic()
    t_first_token = None
    t_first_reasoning = None
    content_parts = []
    reasoning_parts = []
    output_tokens = 0
    error = None

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            buffer = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk

                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.decode("utf-8", errors="replace").split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning and t_first_reasoning is None:
                                t_first_reasoning = time.monotonic()
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            content = delta.get("content", "")
                            if content and t_first_token is None:
                                t_first_token = time.monotonic()
                            if content:
                                content_parts.append(content)
                        usage = data.get("usage")
                        if usage:
                            output_tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        error = str(e)

    t_end = time.monotonic()
    total_content = "".join(content_parts)
    total_reasoning = "".join(reasoning_parts)

    if output_tokens == 0 and (total_content or total_reasoning):
        output_tokens = max(1, len(total_content + total_reasoning) // 4)

    return {
        "ttft": round(t_first_token - t_start, 2) if t_first_token else None,
        "ttfr": round(t_first_reasoning - t_start, 2) if t_first_reasoning else None,
        "total": round(t_end - t_start, 2),
        "output_tokens": output_tokens,
        "content": total_content,
        "reasoning": total_reasoning,
        "error": error,
    }


def run_benchmark():
    """Run cold + hot benchmarks and task-specific eval for all tiers."""
    timing_results = []
    task_results = []

    # Phase 1: Timing benchmarks
    print("=" * 70)
    print("  PHASE 1: TIMING BENCHMARKS (generic prompt)")
    print("=" * 70)

    for tier in TIERS:
        print(f"\n--- {tier} ---")
        ollama_stop_all()

        print("  Cold request...")
        cold = stream_request(tier, GENERIC_PROMPT, MAX_TOKENS)
        if cold["error"]:
            print(f"  ERROR: {cold['error']}")
            timing_results.append({"tier": tier, "error": cold["error"]})
            continue

        ctx = ollama_ps_context()
        print(
            f"  Cold: TTFT={cold['ttft']}s  Total={cold['total']}s  Tokens={cold['output_tokens']}  Ctx={ctx}"
        )

        print("  Hot request...")
        hot = stream_request(tier, GENERIC_PROMPT, MAX_TOKENS)
        hot_toks = (
            hot["output_tokens"] / hot["total"]
            if hot["total"] > 0 and hot["output_tokens"] > 0
            else 0
        )
        print(
            f"  Hot:  TTFT={hot['ttft']}s  Total={hot['total']}s  Tokens={hot['output_tokens']}  tok/s={round(hot_toks, 1)}"
        )

        timing_results.append(
            {
                "tier": tier,
                "ctx": ctx,
                "cold_ttft": cold["ttft"],
                "cold_total": cold["total"],
                "cold_tokens": cold["output_tokens"],
                "hot_ttft": hot["ttft"],
                "hot_total": hot["total"],
                "hot_tokens": hot["output_tokens"],
                "hot_toks": round(hot_toks, 1),
            }
        )

    # Phase 2: Task-specific evaluation
    print("\n" + "=" * 70)
    print("  PHASE 2: TASK-SPECIFIC EVALUATION")
    print("=" * 70)

    for tier in TIERS:
        task = TASK_PROMPTS[tier]
        print(f"\n--- {tier} ---")
        print(f"  Criteria: {task['criteria']}")
        prompt = str(task["prompt"])
        print(f"  Prompt: {prompt[:80]}...")

        # Use hot model if it's still loaded, otherwise cold load
        result = stream_request(tier, prompt, int(task["max_tokens"]))
        if result["error"]:
            print(f"  ERROR: {result['error']}")
            task_results.append({"tier": tier, "error": result["error"]})
            continue

        content = result["content"]
        print(
            f"  TTFT={result['ttft']}s  Total={result['total']}s  Tokens={result['output_tokens']}"
        )
        print(f"  --- Response ({len(content)} chars) ---")
        # Print first 500 chars, truncate rest
        if len(content) > 500:
            print(f"  {content[:500]}...")
            print(f"  [...{len(content) - 500} more chars...]")
        else:
            print(f"  {content}")
        print("  --- End ---")

        task_results.append(
            {
                "tier": tier,
                "ttft": result["ttft"],
                "total": result["total"],
                "tokens": result["output_tokens"],
                "content_length": len(content),
                "content_preview": content[:300],
            }
        )

        # Unload between task evals to ensure clean state for next tier
        ollama_stop_all()

    # Summary
    print("\n" + "=" * 70)
    print("  TIMING SUMMARY")
    print("=" * 70)
    header = f"{'Tier':<14} {'Ctx':>5} {'Cold TTFT':>10} {'Cold Tot':>9} {'Hot TTFT':>9} {'Hot Tot':>8} {'tok/s':>6}"
    print(header)
    print("-" * len(header))
    for r in timing_results:
        if "error" in r and "cold_ttft" not in r:
            print(f"{r['tier']:<14} ERROR: {r.get('error', '')[:40]}")
            continue
        cold_ttft = f"{r['cold_ttft']}s" if r["cold_ttft"] else "-"
        hot_ttft = f"{r['hot_ttft']}s" if r["hot_ttft"] else "-"
        print(
            f"{r['tier']:<14} {r.get('ctx', '?'):>5} "
            f"{cold_ttft:>10} {r['cold_total']:>8}s "
            f"{hot_ttft:>9} {r['hot_total']:>7}s "
            f"{r['hot_toks']:>6}"
        )

    return timing_results, task_results


if __name__ == "__main__":
    run_benchmark()
