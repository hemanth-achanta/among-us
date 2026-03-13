"""
Question complexity estimation.

The ``ComplexityEstimator`` classifies a user's natural-language question into
one of three tiers (LOW / MEDIUM / HIGH) using a combination of:

1. **Heuristic rule-based pass** — fast, free, zero tokens.
2. **LLM-based pass** — slower but more accurate; used when heuristics are
   uncertain or when the question contains ambiguous language.

The heuristic pass is intentionally conservative: it only emits HIGH/LOW with
high confidence; ambiguous cases fall through to the LLM.
"""
from __future__ import annotations

import re

from config.config import ComplexityLevel
from llm.llm_client import LLMClient
from llm.prompt_templates import build_complexity_messages
from utils.logger import Timer, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic signals
# ─────────────────────────────────────────────────────────────────────────────

# Words/phrases that strongly suggest HIGH complexity
_HIGH_SIGNALS: list[str] = [
    r"\bwhy\b", r"\bcause[sd]?\b", r"\bdriv(e|es|ing|er)\b",
    r"\bcompare\b", r"\btrend\b", r"\byear.?over.?year\b", r"\byoy\b",
    r"\bforecast\b", r"\bpredict\b", r"\brank\b", r"\bperformance\b",
    r"\bcohort\b", r"\bsegment\b", r"\bcontribut\b", r"\bbreak.?down\b",
    r"\battribut\b",
]

# Words/phrases that strongly suggest LOW complexity
_LOW_SIGNALS: list[str] = [
    r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bsum\b",
    r"\baverage\b", r"\bavg\b", r"\bmaximum\b", r"\bminimum\b",
    r"\blist\b", r"\bshow me\b", r"\bwhat is the\b", r"\bwhat are the\b",
]

# Structural signals
_JOIN_SIGNALS = re.compile(
    r"\bjoin\b|\bcombine\b|\bmerge\b|\bacross\b|\brelat\b|\blinked\b",
    re.IGNORECASE,
)
_TIME_SIGNALS = re.compile(
    r"\blast (week|month|quarter|year|day)\b"
    r"|\bover time\b|\bmonth.?to.?month\b|\btrending\b",
    re.IGNORECASE,
)
_MULTI_STEP_SIGNALS = re.compile(
    r"\bthen\b|\bafter that\b|\bnext\b|\bfollowed by\b|\bstep\b",
    re.IGNORECASE,
)


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    total = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            total += 1
    return total


def _heuristic_estimate(question: str) -> ComplexityLevel | None:
    """
    Return a complexity level if the heuristics are confident, else None.

    Returns None to signal the caller should fall back to LLM classification.
    """
    high_hits = _count_pattern_hits(question, _HIGH_SIGNALS)
    low_hits  = _count_pattern_hits(question, _LOW_SIGNALS)
    has_join  = bool(_JOIN_SIGNALS.search(question))
    has_time  = bool(_TIME_SIGNALS.search(question))
    has_step  = bool(_MULTI_STEP_SIGNALS.search(question))

    word_count = len(question.split())

    # Definite HIGH
    if has_step or (high_hits >= 2):
        return ComplexityLevel.HIGH

    # Definite MEDIUM
    if has_join or has_time or (high_hits == 1 and low_hits == 0):
        return ComplexityLevel.MEDIUM

    # Definite LOW
    if low_hits >= 1 and not has_join and not has_time and word_count <= 20:
        return ComplexityLevel.LOW

    # Uncertain — delegate to LLM
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ComplexityEstimator
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityEstimator:
    """
    Estimates the analytical complexity of a user question.

    Uses a two-pass strategy:
      1. Fast heuristic rules (no LLM call).
      2. LLM classification if heuristics are uncertain.

    Parameters
    ----------
    llm_client:      Shared :class:`~llm.llm_client.LLMClient` instance.
    schema_summary:  Short schema summary string (from ``SchemaLoader``).
    use_llm_fallback: Set to ``False`` to disable the LLM pass (heuristic only,
                      with MEDIUM as the default for uncertain cases).
    """

    # Model used for complexity estimation (always cheap model — it's a simple task)
    _ESTIMATION_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        llm_client: LLMClient,
        schema_summary: str,
        use_llm_fallback: bool = True,
    ) -> None:
        self._llm            = llm_client
        self._schema_summary = schema_summary
        self._use_llm        = use_llm_fallback

    def estimate(self, question: str) -> ComplexityLevel:
        """
        Classify the question and return a :class:`ComplexityLevel`.

        Parameters
        ----------
        question: Natural-language question from the user.
        """
        with Timer() as t:
            # Pass 1 — heuristics
            heuristic_result = _heuristic_estimate(question)

            if heuristic_result is not None:
                log.info(
                    "complexity_estimated_heuristic",
                    complexity_level=heuristic_result.value,
                    duration_ms=round(t.elapsed_ms, 1),
                )
                return heuristic_result

            # Pass 2 — LLM
            if not self._use_llm:
                log.info(
                    "complexity_estimated_default_medium",
                    reason="llm_fallback_disabled",
                )
                return ComplexityLevel.MEDIUM

            return self._llm_estimate(question)

    # ── LLM-based classification ──────────────────────────────────────────────

    def _llm_estimate(self, question: str) -> ComplexityLevel:
        import json

        messages, system = build_complexity_messages(
            question=question,
            schema_summary=self._schema_summary,
        )

        try:
            with Timer() as t:
                response = self._llm.complete(
                    model=self._ESTIMATION_MODEL,
                    messages=messages,
                    system=system,
                    max_tokens=128,
                )

            parsed = json.loads(response.content)
            raw_level = parsed.get("complexity", "medium").lower()

            level = ComplexityLevel(raw_level)
            log.info(
                "complexity_estimated_llm",
                complexity_level=level.value,
                reasoning=parsed.get("reasoning", ""),
                duration_ms=round(t.elapsed_ms, 1),
            )
            return level

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            log.warning(
                "complexity_estimation_parse_error",
                error=str(exc),
                fallback="medium",
            )
            return ComplexityLevel.MEDIUM
