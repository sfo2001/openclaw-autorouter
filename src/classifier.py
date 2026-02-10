"""Request classification for auto-routing.

Detects task type (vision/coding/reasoning/general) and complexity
(simple/standard/complex) from the last user message. Bilingual DE/EN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    VISION = "vision"
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL = "general"


class Complexity(StrEnum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class LatencyMode(StrEnum):
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    URGENT = "urgent"


@dataclass(frozen=True)
class RoutingDecision:
    task_type: TaskType
    complexity: Complexity
    latency_mode: LatencyMode


# --- Coding detection patterns ---

_CODE_BLOCK_RE = re.compile(r"```")

_CODING_KEYWORDS = re.compile(
    r"\b(?:"
    r"def |function |class |import |from .+ import|"
    r"const |let |var |return |async |await |"
    r"if\s*\(|for\s*\(|while\s*\(|switch\s*\(|"
    r"try\s*\{|catch\s*\(|"
    r"public |private |protected |static |void |"
    r"#include|#define|#ifdef|"
    r"package |interface |struct |enum |"
    r"SELECT |INSERT |UPDATE |DELETE |CREATE TABLE|"
    r"npm |pip |cargo |docker |git |kubectl "
    r")",
    re.IGNORECASE,
)

_ERROR_TRACE_RE = re.compile(
    r"(?:"
    r"Traceback \(most recent call last\)|"
    r"Error:|Exception:|"
    r"at .+\(.+:\d+:\d+\)|"  # JS stack trace
    r"File \".+\", line \d+|"  # Python stack trace
    r"panic:|SIGSEGV|segfault"
    r")",
    re.IGNORECASE,
)

_FILE_PATH_RE = re.compile(
    r"(?:^|\s)(?:"
    r"[./~][a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10}|"  # ./foo/bar.py, ~/config.yaml
    r"[a-zA-Z]:\\[^\s]+|"  # C:\Users\...
    r"/[a-z]+/[a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10}"  # /usr/local/bin/foo.sh
    r")"
)

# --- Reasoning detection patterns ---

_MATH_RE = re.compile(
    r"(?:"
    r"\d+\s*[+\-*/^]\s*\d+\s*=|"  # arithmetic: 2 + 3 =
    r"\\(?:frac|sum|int|lim|sqrt|log|ln)|"  # LaTeX
    r"\b(?:P\(|E\[|Var\()|"  # probability/stats notation
    r"[=<>]\s*\\?[a-z]\s*[+\-*/^]|"  # algebraic expressions
    r"\b\d+\s*(?:mod|%)\s*\d+|"  # modular arithmetic
    r"\bQ\.E\.D\b|"
    r"(?:∀|∃|∈|∉|⊂|⊃|∩|∪|→|↔|¬|∧|∨|⊢|⊨)"  # noqa: RUF001  # logic/set symbols
    r")"
)

_REASONING_KEYWORDS_EN = re.compile(
    r"\b(?:"
    r"prove|derive|proof|theorem|contradiction|induction|"
    r"axiom|corollary|lemma|conjecture|"
    r"if and only if|necessary and sufficient|"
    r"formal logic|propositional|predicate"
    r")\b",
    re.IGNORECASE,
)

_REASONING_KEYWORDS_DE = re.compile(
    r"\b(?:"
    r"beweise|beweisen|beweis|ableiten|herleitung|"
    r"widerspruch|induktion|"
    r"axiom|satz|korollar|vermutung|"
    r"genau dann wenn|hinreichend und notwendig|"
    # Umlaut-free variants (ae/oe/ue)
    r"beweise|ableitung"
    r")\b",
    re.IGNORECASE,
)

# --- Complexity detection patterns ---

_COMPLEXITY_KEYWORDS_EN = re.compile(
    r"\b(?:"
    r"analyze|analyse|compare and contrast|design|"
    r"explain in detail|step by step|in depth|"
    r"elaborate|comprehensive|thorough"
    r")\b",
    re.IGNORECASE,
)

_COMPLEXITY_KEYWORDS_DE = re.compile(
    r"\b(?:"
    r"analysiere|analysieren|vergleiche|vergleichen|"
    r"entwerfe|entwerfen|"
    # Direct unicode umlauts
    r"erkl\u00e4re im detail|erkl\u00e4ren sie im detail|"
    r"schritt f\u00fcr schritt|"
    r"ausf\u00fchrlich|detailliert|"
    # ae/ue/oe umlaut-free variants
    r"erklaere im detail|erklaeren sie im detail|"
    r"schritt fuer schritt|"
    r"ausfuehrlich"
    r")\b",
    re.IGNORECASE,
)

_MULTI_PART_RE = re.compile(
    r"(?:"
    r"^\s*\d+[.)]\s|"  # numbered list: "1." or "1)"
    r"^\s*[-*]\s"  # bullet list
    r")",
    re.MULTILINE,
)

SIMPLE_WORD_THRESHOLD = 50
COMPLEX_WORD_THRESHOLD = 300


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract text content from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
    return ""


def _has_image_content(messages: list[dict[str, Any]]) -> bool:
    """Check if the last user message contains image_url blocks."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
        return False
    return False


def detect_task_type(messages: list[dict[str, Any]]) -> TaskType:
    """Detect task type from messages. Priority: vision > coding > reasoning > general."""
    if _has_image_content(messages):
        return TaskType.VISION

    text = _extract_last_user_text(messages)

    # Coding detection
    if (
        _CODE_BLOCK_RE.search(text)
        or _CODING_KEYWORDS.search(text)
        or _ERROR_TRACE_RE.search(text)
        or _FILE_PATH_RE.search(text)
    ):
        return TaskType.CODING

    # Reasoning detection
    if (
        _MATH_RE.search(text)
        or _REASONING_KEYWORDS_EN.search(text)
        or _REASONING_KEYWORDS_DE.search(text)
    ):
        return TaskType.REASONING

    return TaskType.GENERAL


def detect_complexity(text: str) -> Complexity:
    """Detect complexity level for general task type."""
    word_count = len(text.split())
    has_complexity_keywords = bool(
        _COMPLEXITY_KEYWORDS_EN.search(text) or _COMPLEXITY_KEYWORDS_DE.search(text)
    )
    multi_part_matches = len(_MULTI_PART_RE.findall(text))

    # Complex: long messages, complexity keywords, or multi-part structure
    if word_count > COMPLEX_WORD_THRESHOLD or has_complexity_keywords or multi_part_matches >= 3:
        return Complexity.COMPLEX

    # Simple: short messages without complexity markers
    if word_count < SIMPLE_WORD_THRESHOLD:
        return Complexity.SIMPLE

    return Complexity.STANDARD


def parse_latency_mode(model_name: str) -> LatencyMode:
    """Parse latency mode from model name."""
    if model_name == "auto-urgent":
        return LatencyMode.URGENT
    if model_name == "auto-bg":
        return LatencyMode.BACKGROUND
    return LatencyMode.INTERACTIVE


def classify_request(
    messages: list[dict[str, Any]],
    model_name: str,
) -> RoutingDecision:
    """Classify a chat completion request for routing."""
    latency_mode = parse_latency_mode(model_name)
    task_type = detect_task_type(messages)

    if task_type == TaskType.GENERAL:
        text = _extract_last_user_text(messages)
        complexity = detect_complexity(text)
    else:
        # Non-general task types don't use complexity axis
        complexity = Complexity.STANDARD

    return RoutingDecision(
        task_type=task_type,
        complexity=complexity,
        latency_mode=latency_mode,
    )
