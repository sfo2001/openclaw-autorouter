"""Tests for routing statistics module."""

import time

from src.stats import get_stats, record_auto_route, record_passthrough, reset


class TestStats:
    def setup_method(self):
        reset()

    def test_initial_state(self):
        s = get_stats()
        assert s["total_requests"] == 0
        assert s["auto_requests"] == 0
        assert s["passthrough_requests"] == 0
        assert s["by_task_type"] == {}
        assert s["by_complexity"] == {}
        assert s["by_latency_mode"] == {}
        assert s["by_selected_tier"] == {}
        assert s["by_hot_cold"] == {}
        assert s["by_fallback"] == {}

    def test_uptime(self):
        s = get_stats()
        assert s["uptime_seconds"] >= 0.0
        assert isinstance(s["uptime_seconds"], float)

    def test_record_auto_route(self):
        record_auto_route(
            task_type="general",
            complexity="simple",
            latency_mode="interactive",
            tier="fast",
            was_hot=True,
            was_fallback=False,
        )
        s = get_stats()
        assert s["total_requests"] == 1
        assert s["auto_requests"] == 1
        assert s["passthrough_requests"] == 0
        assert s["by_task_type"] == {"general": 1}
        assert s["by_complexity"] == {"simple": 1}
        assert s["by_latency_mode"] == {"interactive": 1}
        assert s["by_selected_tier"] == {"fast": 1}
        assert s["by_hot_cold"] == {"hot": 1}
        assert s["by_fallback"] == {"ideal": 1}

    def test_record_auto_route_cold_fallback(self):
        record_auto_route(
            task_type="coding",
            complexity="complex",
            latency_mode="background",
            tier="standard",
            was_hot=False,
            was_fallback=True,
        )
        s = get_stats()
        assert s["by_hot_cold"] == {"cold": 1}
        assert s["by_fallback"] == {"fallback": 1}

    def test_record_passthrough(self):
        record_passthrough("complex")
        s = get_stats()
        assert s["total_requests"] == 1
        assert s["auto_requests"] == 0
        assert s["passthrough_requests"] == 1
        assert s["by_selected_tier"] == {"complex": 1}

    def test_multiple_records(self):
        record_auto_route(
            "general",
            "simple",
            "interactive",
            "fast",
            was_hot=True,
            was_fallback=False,
        )
        record_auto_route(
            "coding",
            "standard",
            "interactive",
            "coding-light",
            was_hot=True,
            was_fallback=False,
        )
        record_auto_route(
            "general",
            "complex",
            "interactive",
            "standard",
            was_hot=True,
            was_fallback=True,
        )
        record_passthrough("vision")
        record_passthrough("vision")

        s = get_stats()
        assert s["total_requests"] == 5
        assert s["auto_requests"] == 3
        assert s["passthrough_requests"] == 2
        assert s["by_task_type"] == {"general": 2, "coding": 1}
        assert s["by_selected_tier"]["fast"] == 1
        assert s["by_selected_tier"]["coding-light"] == 1
        assert s["by_selected_tier"]["standard"] == 1
        assert s["by_selected_tier"]["vision"] == 2

    def test_reset(self):
        record_auto_route(
            "general",
            "simple",
            "interactive",
            "fast",
            was_hot=True,
            was_fallback=False,
        )
        record_passthrough("complex")
        reset()

        s = get_stats()
        assert s["total_requests"] == 0
        assert s["auto_requests"] == 0
        assert s["passthrough_requests"] == 0
        assert s["by_task_type"] == {}

    def test_uptime_increases(self):
        s1 = get_stats()
        time.sleep(0.05)
        s2 = get_stats()
        assert s2["uptime_seconds"] >= s1["uptime_seconds"]

    def test_stats_returns_dict_copies(self):
        """Modifying returned dicts should not affect internal state."""
        record_auto_route(
            "general",
            "simple",
            "interactive",
            "fast",
            was_hot=True,
            was_fallback=False,
        )
        s = get_stats()
        s["by_task_type"]["general"] = 999
        s2 = get_stats()
        assert s2["by_task_type"]["general"] == 1
