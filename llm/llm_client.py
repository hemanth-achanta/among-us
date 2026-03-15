"""
Thin, reusable wrapper around the Anthropic Messages API.

Responsibilities
----------------
* Hold a single Anthropic client instance (lazy-initialised, thread-safe).
* Expose a `complete()` method that handles retries on transient API errors.
* Track and expose token usage for cost auditing.
* Raise domain-specific exceptions so callers do not leak Anthropic SDK types.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from config import config
from utils.logger import Timer, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Parsed response from a single LLM call."""

    content:       str
    model:         str
    input_tokens:  int
    output_tokens: int
    stop_reason:   str
    duration_ms:   float
    max_tokens_budget: int = 0
    output_truncated:  bool = False


@dataclass
class TokenUsageSummary:
    """Cumulative token usage tracked across multiple calls in a session."""

    total_input:  int = field(default=0)
    total_output: int = field(default=0)
    call_count:   int = field(default=0)
    # Per-call (model, input_tokens, output_tokens) for cost calculation
    _calls: list[tuple[str, int, int]] = field(default_factory=list)

    def add(self, response: LLMResponse) -> None:
        self.total_input  += response.input_tokens
        self.total_output += response.output_tokens
        self.call_count   += 1
        self._calls.append(
            (response.model, response.input_tokens, response.output_tokens)
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def cost_usd(
        self,
        pricing: dict[str, tuple[float, float]],
        default_per_million: tuple[float, float] = (3.0, 15.0),
    ) -> float:
        """
        Estimate API cost in USD from token usage and per-model pricing.

        pricing: model_id -> (usd_per_1m_input, usd_per_1m_output)
        default_per_million: used when model is not in pricing (e.g. Sonnet).
        """
        total = 0.0
        for model, inp, out in self._calls:
            in_per_m, out_per_m = pricing.get(model, default_per_million)
            total += (inp / 1_000_000.0) * in_per_m + (out / 1_000_000.0) * out_per_m
        return round(total, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when the LLM call fails after all retries are exhausted."""


class LLMRateLimitError(LLMError):
    """Raised specifically on rate-limit exhaustion."""


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thread-safe, singleton-style wrapper around ``anthropic.Anthropic``.

    Instantiate once and share across the application.

    Parameters
    ----------
    api_key:  Anthropic API key (defaults to config value).
    timeout:  Per-request timeout in seconds.
    max_api_retries: How many times to retry on transient API errors (5xx,
                     rate-limit).
    """

    # Exponential back-off constants (seconds)
    _BACKOFF_BASE:   float = 2.0
    _BACKOFF_MAX:    float = 60.0

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        max_api_retries: int = 3,
    ) -> None:
        self._api_key       = api_key or config.ANTHROPIC_API_KEY
        self._timeout       = timeout or config.LLM_REQUEST_TIMEOUT
        self._max_api_retries = max_api_retries
        self._usage         = TokenUsageSummary()

        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured.")

        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=float(self._timeout),
        )

        log.info("llm_client_initialised", timeout=self._timeout)

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        system: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Send a chat-completion request and return an :class:`LLMResponse`.

        Retries on ``anthropic.RateLimitError`` and ``anthropic.InternalServerError``
        with exponential back-off.

        Parameters
        ----------
        model:       Anthropic model identifier.
        messages:    List of ``{"role": ..., "content": ...}`` dicts.
        system:      System prompt string.
        max_tokens:  Cap on output tokens (defaults from ``TOKEN_LIMITS``).
        temperature: Sampling temperature (defaults to ``LLM_TEMPERATURE``).

        Raises
        ------
        LLMRateLimitError: If rate-limit retries are exhausted.
        LLMError:          For all other failure modes.
        """
        _max_tokens  = max_tokens  or config.TOKEN_LIMITS.get(model, 4_096)
        _temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

        last_exc: Exception | None = None

        for attempt in range(1, self._max_api_retries + 1):
            try:
                with Timer() as t:
                    response = self._client.messages.create(
                        model=model,
                        system=system,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=_max_tokens,
                        temperature=_temperature,
                    )

                stop = response.stop_reason or "unknown"
                was_truncated = stop == "max_tokens"

                llm_response = LLMResponse(
                    content=response.content[0].text,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    stop_reason=stop,
                    duration_ms=t.elapsed_ms,
                    max_tokens_budget=_max_tokens,
                    output_truncated=was_truncated,
                )

                self._usage.add(llm_response)

                log.info(
                    "llm_call_success",
                    model=model,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    stop_reason=stop,
                    output_truncated=was_truncated,
                    duration_ms=round(llm_response.duration_ms, 1),
                    attempt=attempt,
                )

                return llm_response

            except anthropic.RateLimitError as exc:
                last_exc = exc
                wait = min(self._BACKOFF_BASE ** attempt, self._BACKOFF_MAX)
                log.warning(
                    "llm_rate_limit",
                    attempt=attempt,
                    max_retries=self._max_api_retries,
                    wait_seconds=wait,
                )
                if attempt < self._max_api_retries:
                    time.sleep(wait)
                else:
                    raise LLMRateLimitError(
                        f"Rate limit exceeded after {self._max_api_retries} attempts."
                    ) from exc

            except anthropic.InternalServerError as exc:
                last_exc = exc
                wait = min(self._BACKOFF_BASE ** attempt, self._BACKOFF_MAX)
                log.warning(
                    "llm_server_error",
                    attempt=attempt,
                    status_code=exc.status_code,
                    wait_seconds=wait,
                )
                if attempt < self._max_api_retries:
                    time.sleep(wait)

            except anthropic.APIError as exc:
                # Non-retriable API errors (bad request, auth, etc.)
                log.error("llm_api_error", error=str(exc))
                raise LLMError(f"LLM API error: {exc}") from exc

        raise LLMError(
            f"LLM call failed after {self._max_api_retries} attempts."
        ) from last_exc

    # ── Token usage ───────────────────────────────────────────────────────────

    @property
    def usage(self) -> TokenUsageSummary:
        """Cumulative token usage since this client was created."""
        return self._usage

    def reset_usage(self) -> None:
        """Reset cumulative usage counters (call at the start of each user session)."""
        self._usage = TokenUsageSummary()
