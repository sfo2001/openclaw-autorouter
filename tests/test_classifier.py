"""Tests for request classifier."""

from typing import Any

from src.classifier import (
    Complexity,
    LatencyMode,
    RoutingDecision,
    TaskType,
    classify_request,
    detect_complexity,
    detect_task_type,
    parse_latency_mode,
)

# --- Vision detection ---


class TestVisionDetection:
    def test_image_url_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        assert detect_task_type(messages) == TaskType.VISION

    def test_text_only_not_vision(self):
        messages = [{"role": "user", "content": "describe an image"}]
        assert detect_task_type(messages) != TaskType.VISION

    def test_image_in_earlier_message_ignored(self):
        """Only the last user message matters."""
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
            {"role": "assistant", "content": "I see a cat."},
            {"role": "user", "content": "Now tell me a joke."},
        ]
        assert detect_task_type(messages) != TaskType.VISION


# --- Coding detection ---


class TestCodingDetection:
    def test_code_block(self):
        messages = [{"role": "user", "content": "Fix this:\n```python\nprint('hello')\n```"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_python_keyword(self):
        messages = [{"role": "user", "content": "How do I use def main() in Python?"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_import_statement(self):
        messages = [{"role": "user", "content": "What does import os do?"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_from_import(self):
        messages = [{"role": "user", "content": "How to use from pathlib import Path"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_traceback(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "I got this error:\n"
                    'Traceback (most recent call last):\n  File "main.py", line 1'
                ),
            }
        ]
        assert detect_task_type(messages) == TaskType.CODING

    def test_js_stack_trace(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "Error: Cannot read property\n    at Object.<anonymous> (index.js:10:5)"
                ),
            }
        ]
        assert detect_task_type(messages) == TaskType.CODING

    def test_file_path(self):
        messages = [{"role": "user", "content": "Edit the file ./src/main.py please"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_docker_command(self):
        messages = [{"role": "user", "content": "How do I run docker build -t myapp ."}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_sql_keyword(self):
        messages = [{"role": "user", "content": "Write a SELECT query for users table"}]
        assert detect_task_type(messages) == TaskType.CODING

    def test_no_coding_markers(self):
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        assert detect_task_type(messages) != TaskType.CODING


# --- Reasoning detection ---


class TestReasoningDetection:
    def test_proof_keyword_en(self):
        messages = [{"role": "user", "content": "Prove that sqrt(2) is irrational"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_theorem_keyword_en(self):
        messages = [{"role": "user", "content": "Explain the Pythagorean theorem and its proof"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_derive_keyword_en(self):
        messages = [
            {"role": "user", "content": "Derive the quadratic formula from ax^2 + bx + c = 0"}
        ]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_contradiction_en(self):
        messages = [
            {
                "role": "user",
                "content": "Show by contradiction that there are infinitely many primes",
            }
        ]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_induction_en(self):
        messages = [{"role": "user", "content": "Prove by induction that 1+2+...+n = n(n+1)/2"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_beweis_de(self):
        messages = [{"role": "user", "content": "Beweise, dass die Wurzel aus 2 irrational ist"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_ableiten_de(self):
        messages = [
            {"role": "user", "content": "Leite die quadratische Formel ab. Ableiten bitte."}
        ]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_widerspruch_de(self):
        messages = [
            {
                "role": "user",
                "content": "Zeige per Widerspruch, dass es unendlich viele Primzahlen gibt",
            }
        ]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_induktion_de(self):
        messages = [
            {"role": "user", "content": "Beweise per Induktion, dass 1+2+...+n = n(n+1)/2"}
        ]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_math_notation(self):
        messages = [{"role": "user", "content": "Solve: \\frac{x^2 + 3x}{2} = 5"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_logic_symbols(self):
        messages = [{"role": "user", "content": "Show that ∀x ∈ S: P(x) → Q(x)"}]
        assert detect_task_type(messages) == TaskType.REASONING

    def test_no_reasoning_markers(self):
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        # Simple arithmetic without proof/derive keywords -> not reasoning
        assert detect_task_type(messages) != TaskType.REASONING


# --- Coding takes priority over reasoning ---


class TestPriority:
    def test_code_with_math(self):
        """Code blocks take priority over math notation."""
        messages = [
            {
                "role": "user",
                "content": (
                    "```python\nimport math\nprint(math.sqrt(2))\n```\nProve this converges"
                ),
            }
        ]
        assert detect_task_type(messages) == TaskType.CODING

    def test_vision_over_coding(self):
        """Image takes priority over code keywords."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does this function do?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        assert detect_task_type(messages) == TaskType.VISION


# --- Complexity detection ---


class TestComplexityDetection:
    def test_simple_short(self):
        assert detect_complexity("Hello, how are you?") == Complexity.SIMPLE

    def test_simple_greeting(self):
        assert detect_complexity("Hi") == Complexity.SIMPLE

    def test_simple_factual(self):
        assert detect_complexity("What is the capital of France?") == Complexity.SIMPLE

    def test_standard_medium(self):
        text = "Explain how neural networks work and what backpropagation does. " * 10
        assert detect_complexity(text) == Complexity.STANDARD

    def test_complex_long(self):
        text = "word " * 350  # >300 words
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_en_analyze(self):
        text = "Please analyze the economic impact of trade policy changes on small businesses."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_en_step_by_step(self):
        text = "Explain step by step how a compiler transforms source code into machine code."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_en_compare(self):
        text = "Compare and contrast the approaches used by React and Vue for state management."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_de_analysiere(self):
        text = "Analysiere die Auswirkungen der Handelspolitik auf kleine Unternehmen."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_de_schritt_fuer_schritt(self):
        text = "Erkl\u00e4re Schritt f\u00fcr Schritt, wie ein Compiler funktioniert."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_de_ausfuehrlich(self):
        text = "Beschreibe ausfuehrlich die Geschichte der Relativitaetstheorie."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_keyword_de_detailliert(self):
        text = "Erkl\u00e4re detailliert, wie das Immunsystem funktioniert."
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_complex_multi_part(self):
        text = (
            "Please address these points:\n"
            "1. What is the current state?\n"
            "2. What are the risks?\n"
            "3. What should we do next?\n"
        )
        assert detect_complexity(text) == Complexity.COMPLEX

    def test_short_with_complexity_keyword_is_complex(self):
        """Complexity keywords override word count -- even short text -> complex."""
        text = "Analyze this."
        assert detect_complexity(text) == Complexity.COMPLEX


# --- Latency mode parsing ---


class TestLatencyModeParsing:
    def test_auto(self):
        assert parse_latency_mode("auto") == LatencyMode.INTERACTIVE

    def test_auto_bg(self):
        assert parse_latency_mode("auto-bg") == LatencyMode.BACKGROUND

    def test_auto_urgent(self):
        assert parse_latency_mode("auto-urgent") == LatencyMode.URGENT

    def test_unknown_defaults_interactive(self):
        assert parse_latency_mode("something-else") == LatencyMode.INTERACTIVE


# --- Full classify_request ---


class TestClassifyRequest:
    def test_vision_interactive(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        result = classify_request(messages, "auto")
        assert result == RoutingDecision(
            TaskType.VISION, Complexity.STANDARD, LatencyMode.INTERACTIVE
        )

    def test_coding_background(self):
        messages = [{"role": "user", "content": "```python\nprint('hello')\n```"}]
        result = classify_request(messages, "auto-bg")
        assert result == RoutingDecision(
            TaskType.CODING, Complexity.STANDARD, LatencyMode.BACKGROUND
        )

    def test_simple_general_urgent(self):
        messages = [{"role": "user", "content": "Hi there"}]
        result = classify_request(messages, "auto-urgent")
        assert result == RoutingDecision(TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.URGENT)

    def test_complex_general(self):
        messages = [
            {
                "role": "user",
                "content": "Analyze the long-term economic effects of automation on employment. "
                "Compare and contrast different perspectives.",
            }
        ]
        result = classify_request(messages, "auto")
        assert result == RoutingDecision(
            TaskType.GENERAL, Complexity.COMPLEX, LatencyMode.INTERACTIVE
        )

    def test_empty_messages(self):
        result = classify_request([], "auto")
        assert result == RoutingDecision(
            TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.INTERACTIVE
        )

    def test_system_message_only(self):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        result = classify_request(messages, "auto")
        assert result == RoutingDecision(
            TaskType.GENERAL, Complexity.SIMPLE, LatencyMode.INTERACTIVE
        )
