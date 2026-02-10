"""Tests for model selection router."""

import pytest

from src.classifier import Complexity, LatencyMode, TaskType
from src.router import RoutingResult, _loaded_tiers, select_model

# --- Urgent mode: always fast ---


class TestUrgentMode:
    @pytest.mark.parametrize(
        "task_type,complexity",
        [
            (TaskType.VISION, Complexity.STANDARD),
            (TaskType.CODING, Complexity.STANDARD),
            (TaskType.REASONING, Complexity.STANDARD),
            (TaskType.GENERAL, Complexity.SIMPLE),
            (TaskType.GENERAL, Complexity.STANDARD),
            (TaskType.GENERAL, Complexity.COMPLEX),
        ],
    )
    def test_urgent_always_fast(self, task_type, complexity):
        result = select_model(task_type, complexity, LatencyMode.URGENT, [])
        assert result.tier == "fast", f"Expected fast for urgent {task_type}/{complexity}"
        assert not result.was_fallback
        assert result.ideal_tier == "fast"

    def test_urgent_hot_when_loaded(self):
        result = select_model(
            TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.URGENT, ["qwen3-32k-fast:latest"]
        )
        assert result.tier == "fast"
        assert result.was_hot

    def test_urgent_cold_when_not_loaded(self):
        result = select_model(TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.URGENT, [])
        assert result.tier == "fast"
        assert not result.was_hot


# --- Background mode: best quality ---


class TestBackgroundMode:
    def test_vision_bg(self):
        result = select_model(TaskType.VISION, Complexity.STANDARD, LatencyMode.BACKGROUND, [])
        assert result.tier == "vision"
        assert not result.was_fallback
        assert result.ideal_tier == "vision"

    def test_coding_bg(self):
        result = select_model(TaskType.CODING, Complexity.STANDARD, LatencyMode.BACKGROUND, [])
        assert result.tier == "coding"
        assert not result.was_fallback

    def test_reasoning_bg(self):
        result = select_model(TaskType.REASONING, Complexity.STANDARD, LatencyMode.BACKGROUND, [])
        assert result.tier == "reasoning"
        assert not result.was_fallback

    def test_general_simple_bg(self):
        result = select_model(TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.BACKGROUND, [])
        assert result.tier == "fast"

    def test_general_standard_bg(self):
        result = select_model(TaskType.GENERAL, Complexity.STANDARD, LatencyMode.BACKGROUND, [])
        assert result.tier == "standard"

    def test_general_complex_bg(self):
        result = select_model(TaskType.GENERAL, Complexity.COMPLEX, LatencyMode.BACKGROUND, [])
        assert result.tier == "complex"

    def test_bg_hot_when_loaded(self):
        result = select_model(
            TaskType.CODING, Complexity.STANDARD, LatencyMode.BACKGROUND, ["qwen3-coder-next"]
        )
        assert result.tier == "coding"
        assert result.was_hot

    def test_bg_cold_when_not_loaded(self):
        result = select_model(TaskType.CODING, Complexity.STANDARD, LatencyMode.BACKGROUND, [])
        assert result.tier == "coding"
        assert not result.was_hot


# --- Interactive mode: prefer loaded models ---


class TestInteractiveMode:
    def test_coding_nothing_loaded(self):
        """No models loaded -> ideal tier (coding-light for interactive coding)."""
        result = select_model(TaskType.CODING, Complexity.STANDARD, LatencyMode.INTERACTIVE, [])
        assert result.tier == "coding-light"
        assert not result.was_hot
        assert not result.was_fallback
        assert result.ideal_tier == "coding-light"

    def test_coding_coding_loaded(self):
        """Full coding model loaded -> use it (it's in the chain)."""
        result = select_model(
            TaskType.CODING,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["qwen3-coder-next:latest"],
        )
        assert result.tier == "coding"
        assert result.was_hot
        assert result.was_fallback
        assert result.ideal_tier == "coding-light"

    def test_coding_light_loaded(self):
        """coding-light loaded -> prefer it over cold-loading coding."""
        result = select_model(
            TaskType.CODING,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["devstral-small-official-128k:latest"],
        )
        assert result.tier == "coding-light"
        assert result.was_hot
        assert not result.was_fallback

    def test_coding_fast_loaded_only(self):
        """Only fast loaded -> fall back to fast for coding."""
        result = select_model(
            TaskType.CODING,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["qwen3-32k-fast:latest"],
        )
        assert result.tier == "fast"
        assert result.was_hot
        assert result.was_fallback
        assert result.ideal_tier == "coding-light"

    def test_reasoning_nothing_loaded(self):
        """No models loaded -> ideal tier (reasoning)."""
        result = select_model(TaskType.REASONING, Complexity.STANDARD, LatencyMode.INTERACTIVE, [])
        assert result.tier == "reasoning"
        assert not result.was_hot
        assert not result.was_fallback

    def test_reasoning_standard_loaded(self):
        """Standard loaded, reasoning (32B) not loaded -> fall back to standard."""
        result = select_model(
            TaskType.REASONING,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["cogito:14b"],
        )
        assert result.tier == "standard"
        assert result.was_hot
        assert result.was_fallback
        assert result.ideal_tier == "reasoning"

    def test_reasoning_32b_loaded(self):
        """32B loaded -> use reasoning."""
        result = select_model(
            TaskType.REASONING,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["cogito:32b"],
        )
        assert result.tier == "reasoning"
        assert result.was_hot
        assert not result.was_fallback

    def test_complex_general_standard_loaded(self):
        """Complex general but only standard loaded -> fall back to standard."""
        result = select_model(
            TaskType.GENERAL,
            Complexity.COMPLEX,
            LatencyMode.INTERACTIVE,
            ["cogito:14b"],
        )
        assert result.tier == "standard"
        assert result.was_hot
        assert result.was_fallback
        assert result.ideal_tier == "complex"

    def test_complex_general_32b_loaded(self):
        """32B loaded -> use complex."""
        result = select_model(
            TaskType.GENERAL,
            Complexity.COMPLEX,
            LatencyMode.INTERACTIVE,
            ["cogito:32b"],
        )
        assert result.tier == "complex"
        assert result.was_hot
        assert not result.was_fallback

    def test_vision_loaded(self):
        result = select_model(
            TaskType.VISION,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["qwen3-vl:8b-instruct-q4_K_M"],
        )
        assert result.tier == "vision"
        assert result.was_hot
        assert not result.was_fallback

    def test_vision_not_loaded_fast_loaded(self):
        """Vision model not loaded but fast is -> fall back to fast."""
        result = select_model(
            TaskType.VISION,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["qwen3-32k-fast:latest"],
        )
        assert result.tier == "fast"
        assert result.was_hot
        assert result.was_fallback
        assert result.ideal_tier == "vision"

    def test_general_simple_always_fast(self):
        """Simple general is always fast regardless of loaded state."""
        result = select_model(
            TaskType.GENERAL,
            Complexity.SIMPLE,
            LatencyMode.INTERACTIVE,
            ["cogito:32b"],
        )
        assert result.tier == "fast"

    def test_general_standard_loaded(self):
        result = select_model(
            TaskType.GENERAL,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            ["cogito:14b"],
        )
        assert result.tier == "standard"
        assert result.was_hot

    def test_general_standard_nothing_loaded(self):
        result = select_model(
            TaskType.GENERAL,
            Complexity.STANDARD,
            LatencyMode.INTERACTIVE,
            [],
        )
        assert result.tier == "standard"
        assert not result.was_hot
        assert not result.was_fallback


# --- RoutingResult dataclass ---


class TestRoutingResult:
    def test_frozen(self):
        r = RoutingResult(tier="fast", was_hot=True, was_fallback=False, ideal_tier="fast")
        with pytest.raises(AttributeError):
            r.tier = "standard"  # type: ignore[misc]

    def test_equality(self):
        a = RoutingResult(tier="fast", was_hot=True, was_fallback=False, ideal_tier="fast")
        b = RoutingResult(tier="fast", was_hot=True, was_fallback=False, ideal_tier="fast")
        assert a == b


# --- Loaded tiers helper ---


class TestLoadedTiers:
    def test_empty(self):
        assert _loaded_tiers([]) == set()

    def test_strips_latest(self):
        tiers = _loaded_tiers(["qwen3-32k-fast:latest"])
        assert "fast" in tiers

    def test_keeps_explicit_tag(self):
        tiers = _loaded_tiers(["cogito:14b"])
        assert "standard" in tiers

    def test_multi_tier_model(self):
        """cogito:32b serves both complex and reasoning tiers."""
        tiers = _loaded_tiers(["cogito:32b"])
        assert "complex" in tiers
        assert "reasoning" in tiers

    def test_unknown_model_ignored(self):
        tiers = _loaded_tiers(["unknown-model:latest"])
        assert tiers == set()
