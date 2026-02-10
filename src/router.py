"""Model selection based on task type, complexity, and latency mode.

Maps classification results to concrete LiteLLM model tier names.
Interactive mode prefers already-loaded models to avoid cold-load penalties.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.classifier import Complexity, LatencyMode, TaskType

# Tier name -> Ollama model name (must match config.yaml litellm_params.model)
# Stripped of the ollama_chat/ prefix since Ollama /api/ps returns bare names.
TIER_TO_OLLAMA_MODEL: dict[str, str] = {
    "fast": "qwen3-32k-fast",
    "standard": "cogito:14b",
    "complex": "cogito:32b",
    "coding": "qwen3-coder-next",
    "coding-light": "devstral-small-official-128k",
    "thinking": "qwen3-32k",
    "reasoning": "cogito:32b",
    "vision": "qwen3-vl:8b-instruct-q4_K_M",
}

# Reverse: Ollama model name -> set of tier names (some models serve multiple tiers)
_OLLAMA_MODEL_TO_TIERS: dict[str, set[str]] = {}
for _tier, _model in TIER_TO_OLLAMA_MODEL.items():
    _OLLAMA_MODEL_TO_TIERS.setdefault(_model, set()).add(_tier)

# Background mode: best tier for each task type (quality over latency)
_BACKGROUND_MAP: dict[tuple[TaskType, Complexity], str] = {
    (TaskType.VISION, Complexity.SIMPLE): "vision",
    (TaskType.VISION, Complexity.STANDARD): "vision",
    (TaskType.VISION, Complexity.COMPLEX): "vision",
    (TaskType.CODING, Complexity.SIMPLE): "coding",
    (TaskType.CODING, Complexity.STANDARD): "coding",
    (TaskType.CODING, Complexity.COMPLEX): "coding",
    (TaskType.REASONING, Complexity.SIMPLE): "reasoning",
    (TaskType.REASONING, Complexity.STANDARD): "reasoning",
    (TaskType.REASONING, Complexity.COMPLEX): "reasoning",
    (TaskType.GENERAL, Complexity.SIMPLE): "fast",
    (TaskType.GENERAL, Complexity.STANDARD): "standard",
    (TaskType.GENERAL, Complexity.COMPLEX): "complex",
}

# Interactive mode: ideal tier + fallback chain (lighter alternatives)
_INTERACTIVE_MAP: dict[tuple[TaskType, Complexity], list[str]] = {
    (TaskType.VISION, Complexity.SIMPLE): ["vision", "fast"],
    (TaskType.VISION, Complexity.STANDARD): ["vision", "fast"],
    (TaskType.VISION, Complexity.COMPLEX): ["vision", "fast"],
    (TaskType.CODING, Complexity.SIMPLE): ["coding-light", "coding", "fast"],
    (TaskType.CODING, Complexity.STANDARD): ["coding-light", "coding", "fast"],
    (TaskType.CODING, Complexity.COMPLEX): ["coding-light", "coding", "fast"],
    (TaskType.REASONING, Complexity.SIMPLE): ["reasoning", "standard", "fast"],
    (TaskType.REASONING, Complexity.STANDARD): ["reasoning", "standard", "fast"],
    (TaskType.REASONING, Complexity.COMPLEX): ["reasoning", "standard", "fast"],
    (TaskType.GENERAL, Complexity.SIMPLE): ["fast"],
    (TaskType.GENERAL, Complexity.STANDARD): ["standard", "fast"],
    (TaskType.GENERAL, Complexity.COMPLEX): ["complex", "standard", "fast"],
}


@dataclass(frozen=True)
class RoutingResult:
    """Result of model selection with metadata for logging and stats."""

    tier: str
    was_hot: bool
    was_fallback: bool
    ideal_tier: str


def _loaded_tiers(loaded_ollama_models: list[str]) -> set[str]:
    """Convert Ollama model names to the set of tier names currently loaded.

    Ollama /api/ps returns names with optional `:latest` suffix. We strip
    the `:latest` suffix but keep explicit tags like `:14b`.
    """
    tiers: set[str] = set()
    for model_name in loaded_ollama_models:
        # Ollama may return "cogito:14b" or "qwen3-32k-fast:latest"
        bare = model_name.removesuffix(":latest")
        for tier in _OLLAMA_MODEL_TO_TIERS.get(bare, set()):
            tiers.add(tier)
        # Also try the full name with tag
        for tier in _OLLAMA_MODEL_TO_TIERS.get(model_name, set()):
            tiers.add(tier)
    return tiers


def select_model(
    task_type: TaskType,
    complexity: Complexity,
    latency_mode: LatencyMode,
    loaded_ollama_models: list[str],
) -> RoutingResult:
    """Select the best model tier given classification and loaded models.

    Returns a RoutingResult with the selected tier and metadata about the decision.
    """
    key = (task_type, complexity)
    loaded = _loaded_tiers(loaded_ollama_models)

    # Urgent: always fast
    if latency_mode == LatencyMode.URGENT:
        tier = "fast"
        return RoutingResult(
            tier=tier,
            was_hot=tier in loaded,
            was_fallback=False,
            ideal_tier=tier,
        )

    # Background: always best quality (no fallback)
    if latency_mode == LatencyMode.BACKGROUND:
        tier = _BACKGROUND_MAP[key]
        return RoutingResult(
            tier=tier,
            was_hot=tier in loaded,
            was_fallback=False,
            ideal_tier=tier,
        )

    # Interactive: prefer loaded models, fall back to lighter tiers
    chain = _INTERACTIVE_MAP[key]
    ideal_tier = chain[0]

    # First pass: pick the first tier in the preference chain that's loaded
    for tier in chain:
        if tier in loaded:
            return RoutingResult(
                tier=tier,
                was_hot=True,
                was_fallback=tier != ideal_tier,
                ideal_tier=ideal_tier,
            )

    # Nothing loaded: return the first (ideal) tier, accepting cold-load
    return RoutingResult(
        tier=ideal_tier,
        was_hot=False,
        was_fallback=False,
        ideal_tier=ideal_tier,
    )
