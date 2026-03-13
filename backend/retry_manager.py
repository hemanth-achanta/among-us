"""
Retry logic for the SQL generation → validation → execution pipeline.

``RetryManager`` decides:
  1. Whether a retry is warranted given the current failure reason.
  2. Whether to escalate to a stronger model before retrying.
  3. When to give up and surface an error to the orchestrator.

Retry conditions
----------------
RETRY_ON_EXECUTION_ERROR  — Database reported an error (bad SQL syntax, etc.)
RETRY_ON_EMPTY_RESULT     — Query succeeded but returned 0 rows (configurable).
RETRY_ON_LOW_CONFIDENCE   — LLM's own confidence score is below the threshold.
RETRY_ON_VALIDATION_ERROR — SQLValidator rejected the query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from config import config
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations & data classes
# ─────────────────────────────────────────────────────────────────────────────

class RetryReason(Enum):
    EXECUTION_ERROR   = auto()
    EMPTY_RESULT      = auto()
    LOW_CONFIDENCE    = auto()
    VALIDATION_ERROR  = auto()


@dataclass
class RetryState:
    """Mutable state tracking all retries for a single user question."""

    attempt:       int         = 0          # incremented before each retry
    current_model: str         = ""
    reasons:       list[str]   = field(default_factory=list)
    exhausted:     bool        = False      # True when max retries reached

    def record_failure(self, reason: str) -> None:
        self.reasons.append(f"attempt {self.attempt}: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class RetryManager:
    """
    Manages retry decisions and model escalation for the SQL pipeline.

    Parameters
    ----------
    model_router:          :class:`~backend.model_router.ModelRouter` instance.
    initial_model:         The model chosen for the first attempt.
    max_retries:           Hard cap on total retry attempts.
    escalation_threshold:  Escalate to next model after this many failures.
    min_confidence:        Confidence score below which a retry is triggered.
    retry_on_empty:        Whether empty results should trigger a retry.
    """

    def __init__(
        self,
        model_router,                               # avoid circular import
        initial_model:        str,
        max_retries:          int   = config.MAX_RETRIES,
        escalation_threshold: int   = config.RETRY_ESCALATION_THRESHOLD,
        min_confidence:       float = config.MIN_CONFIDENCE_THRESHOLD,
        retry_on_empty:       bool  = config.RETRY_ON_EMPTY_RESULT,
    ) -> None:
        self._router               = model_router
        self._max_retries          = max_retries
        self._escalation_threshold = escalation_threshold
        self._min_confidence       = min_confidence
        self._retry_on_empty       = retry_on_empty

        self._state = RetryState(
            attempt=0,
            current_model=initial_model,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> RetryState:
        """Current retry state (read-only view)."""
        return self._state

    @property
    def current_model(self) -> str:
        """Model to use for the next attempt."""
        return self._state.current_model

    def should_retry(
        self,
        reason: RetryReason,
        detail: str = "",
    ) -> bool:
        """
        Evaluate whether a retry should be attempted and update internal state.

        Call this after each failed pipeline cycle.  If the method returns
        ``True``, the caller should re-run the cycle using ``current_model``.

        Parameters
        ----------
        reason: Why the current attempt failed.
        detail: Optional human-readable error message (for logging).

        Returns
        -------
        ``True`` if a retry should be attempted, ``False`` to give up.
        """
        # Never retry if we're not configured to do so for empty results
        if reason == RetryReason.EMPTY_RESULT and not self._retry_on_empty:
            log.info(
                "retry_skipped",
                reason="empty_result_retry_disabled",
            )
            return False

        self._state.attempt += 1
        failure_msg = f"{reason.name}: {detail}" if detail else reason.name
        self._state.record_failure(failure_msg)

        if self._state.attempt > self._max_retries:
            self._state.exhausted = True
            log.warning(
                "retry_exhausted",
                attempt=self._state.attempt,
                max_retries=self._max_retries,
                reasons=self._state.reasons,
            )
            return False

        # Model escalation
        if self._state.attempt >= self._escalation_threshold:
            next_model = self._router.escalate(self._state.current_model)
            if next_model and next_model != self._state.current_model:
                log.info(
                    "model_escalated_on_retry",
                    attempt=self._state.attempt,
                    from_model=self._state.current_model,
                    to_model=next_model,
                    reason=reason.name,
                )
                self._state.current_model = next_model

        log.info(
            "retry_scheduled",
            attempt=self._state.attempt,
            model=self._state.current_model,
            reason=reason.name,
        )
        return True

    def needs_retry_for_confidence(self, confidence: float) -> bool:
        """
        Return ``True`` if the confidence score warrants a retry.

        Does **not** consume a retry slot — call :meth:`should_retry` with
        ``RetryReason.LOW_CONFIDENCE`` if you decide to proceed.
        """
        return confidence < self._min_confidence

    def reset(self, model: str) -> None:
        """
        Reset state for a new question.

        Parameters
        ----------
        model: The initial model to use for the new question.
        """
        self._state = RetryState(attempt=0, current_model=model)
