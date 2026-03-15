"""
LLM-powered Text-to-SQL generator.

``SQLGenerator`` sends a structured prompt to the LLM and parses the returned
JSON into a typed ``SQLGenerationResult``.  It does NOT execute SQL — that is
the responsibility of ``SQLExecutor``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm.llm_client import LLMClient, LLMError
from llm.prompt_templates import build_sql_generation_messages
from config import config
from utils.logger import Timer, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SQLGenerationResult:
    """Structured output from a single SQL generation call."""

    sql_query:        str | None       # None if the LLM decided the question is unanswerable
    reasoning:        str
    confidence:       float            # 0.0–1.0
    model_used:       str
    raw_content:      str              # Raw LLM response for debugging
    duration_ms:      float
    prompt_system:    str              # System prompt sent to the LLM
    prompt_messages:  list[dict[str, Any]]  # User/assistant messages sent to the LLM
    stop_reason:      str = "unknown"  # "end_turn" = complete, "max_tokens" = truncated
    output_truncated: bool = False
    input_tokens:     int = 0
    output_tokens:    int = 0
    max_tokens_budget: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class SQLGenerator:
    """
    Translates a natural-language question into a SQL query via LLM.

    Parameters
    ----------
    llm_client:  Shared :class:`~llm.llm_client.LLMClient` instance.
    schema_str:  Formatted schema string from ``SchemaLoader.format_for_prompt()``.
    dialect:     SQL dialect string, e.g. "PostgreSQL".
    max_rows:    Row limit injected into the prompt (from config).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        schema_str: str,
        dialect: str,
        max_rows: int = config.MAX_RESULT_ROWS,
    ) -> None:
        self._llm       = llm_client
        self._schema    = schema_str
        self._dialect   = dialect
        self._max_rows  = max_rows

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        model: str,
        previous_queries: list[dict[str, Any]] | None = None,
        chat_history: list[dict[str, str]] | None = None,
        subquery_description: str | None = None,
        schema_override: str | None = None,
        last_failed_attempt: dict[str, str] | None = None,
    ) -> SQLGenerationResult:
        """
        Generate SQL for the given question using the specified model.

        Parameters
        ----------
        question:         Natural-language question from the user.
        model:            Anthropic model ID to use.
        previous_queries: Optional list of ``{"sql": ..., "result_summary": ...}``
                          dicts from previous iterations in this session.

        Returns
        -------
        :class:`SQLGenerationResult`

        Raises
        ------
        LLMError: If the LLM call fails after retries.
        """
        schema_for_prompt = schema_override or self._schema

        messages, system = build_sql_generation_messages(
            question=question,
            schema=schema_for_prompt,
            dialect=self._dialect,
            max_rows=self._max_rows,
            previous_queries=previous_queries,
            chat_history=chat_history,
            subquery_description=subquery_description,
            last_failed_attempt=last_failed_attempt,
        )

        log.info(
            "sql_generation_start",
            model=model,
            question_length=len(question),
            previous_query_count=len(previous_queries or []),
        )

        with Timer() as t:
            response = self._llm.complete(
                model=model,
                messages=messages,
                system=system,
            )

        result = self._parse_response(
            raw=response.content,
            model=model,
            duration_ms=t.elapsed_ms,
            prompt_system=system,
            prompt_messages=messages,
            stop_reason=response.stop_reason,
            output_truncated=response.output_truncated,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            max_tokens_budget=response.max_tokens_budget,
        )

        log.info(
            "sql_generation_complete",
            model=model,
            confidence=result.confidence,
            sql_null=result.sql_query is None,
            stop_reason=response.stop_reason,
            output_truncated=response.output_truncated,
            duration_ms=round(t.elapsed_ms, 1),
        )

        return result

    # ── Response parsing ──────────────────────────────────────────────────────

    # Regex to detect markdown SQL code block: ```sql ... ``` or ```\n...SELECT... ```
    _SQL_BLOCK_RE = re.compile(
        r"```(?:sql)?\s*\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    @staticmethod
    def _extract_sql_from_code_block(raw: str) -> str | None:
        """
        If the raw LLM response contains a markdown SQL code block (e.g. ```sql ... ```),
        return the extracted SQL string; otherwise return None.
        If multiple code blocks exist, returns the first whose content looks like SQL.
        """
        text = raw.strip()
        for match in SQLGenerator._SQL_BLOCK_RE.finditer(text):
            sql = match.group(1).strip()
            if sql.upper().startswith(("SELECT", "WITH")):
                return sql
        return None

    @staticmethod
    def _parse_response(
        raw: str,
        model: str,
        duration_ms: float,
        prompt_system: str,
        prompt_messages: list[dict[str, Any]],
        stop_reason: str = "unknown",
        output_truncated: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        max_tokens_budget: int = 0,
    ) -> SQLGenerationResult:
        """
        Parse the LLM response into a :class:`SQLGenerationResult`.

        Handles three cases:
          1. Clean JSON response (happy path).
          2. JSON embedded inside markdown code fences.
          3. Malformed response — returns a null-SQL result with confidence 0.
        """
        text = raw.strip()

        # ── Strip markdown fences if present ────────────────────────────────
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )
        if fence_match:
            text = fence_match.group(1)
        else:
            # Try to extract bare JSON object
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        llm_metadata = dict(
            stop_reason=stop_reason,
            output_truncated=output_truncated,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_tokens_budget=max_tokens_budget,
        )

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fallback: try to extract SQL from a markdown code block (e.g. ```sql ... ```)
            fallback = SQLGenerator._extract_sql_from_code_block(raw)
            if fallback is not None:
                log.warning(
                    "sql_generation_parse_fallback",
                    error=str(exc),
                    message="Extracted SQL from code block (LLM did not return valid JSON)",
                )
                return SQLGenerationResult(
                    sql_query=fallback,
                    reasoning="Extracted from code block (LLM did not return valid JSON).",
                    confidence=0.7,
                    model_used=model,
                    raw_content=raw,
                    duration_ms=duration_ms,
                    prompt_system=prompt_system,
                    prompt_messages=prompt_messages,
                    **llm_metadata,
                )
            log.warning(
                "sql_generation_parse_error",
                error=str(exc),
                raw_snippet=raw[:300],
            )
            return SQLGenerationResult(
                sql_query=None,
                reasoning=f"Could not parse LLM response: {exc}",
                confidence=0.0,
                model_used=model,
                raw_content=raw,
                duration_ms=duration_ms,
                prompt_system=prompt_system,
                prompt_messages=prompt_messages,
                **llm_metadata,
            )

        sql_query  = parsed.get("sql_query")
        reasoning  = str(parsed.get("reasoning", ""))
        confidence = float(parsed.get("confidence", 0.5))

        # Normalise None-like strings
        if isinstance(sql_query, str) and sql_query.strip().lower() in ("null", "none", ""):
            sql_query = None

        # If JSON had no valid SQL but raw response has a ```sql ... ``` block, use it
        if sql_query is None:
            fallback = SQLGenerator._extract_sql_from_code_block(raw)
            if fallback is not None:
                log.warning(
                    "sql_generation_parse_fallback",
                    message="Extracted SQL from code block (LLM did not return valid JSON).",
                )
                return SQLGenerationResult(
                    sql_query=fallback,
                    reasoning="Extracted from code block (LLM did not return valid JSON).",
                    confidence=0.7,
                    model_used=model,
                    raw_content=raw,
                    duration_ms=duration_ms,
                    prompt_system=prompt_system,
                    prompt_messages=prompt_messages,
                    **llm_metadata,
                )

        return SQLGenerationResult(
            sql_query=sql_query,
            reasoning=reasoning,
            confidence=min(max(confidence, 0.0), 1.0),
            model_used=model,
            raw_content=raw,
            duration_ms=duration_ms,
            prompt_system=prompt_system,
            prompt_messages=prompt_messages,
            **llm_metadata,
        )
