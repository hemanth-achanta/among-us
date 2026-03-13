"""
Model routing: maps question complexity to the appropriate LLM.

``ModelRouter`` also manages the escalation chain used by ``RetryManager``
when a lower-tier model repeatedly fails to produce a valid SQL query.
"""
from __future__ import annotations

from config import config
from config.config import ComplexityLevel
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Routing table
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEXITY_TO_MODEL: dict[ComplexityLevel, str] = {
    ComplexityLevel.LOW:    config.LOW_MODEL,
    ComplexityLevel.MEDIUM: config.MEDIUM_MODEL,
    ComplexityLevel.HIGH:   config.HIGH_MODEL,
}


class ModelRouter:
    """
    Selects the most cost-efficient LLM model for a given complexity level
    and manages the model escalation chain for retry scenarios.

    Parameters
    ----------
    escalation_chain:
        Ordered list of model IDs from weakest to strongest.
        Defaults to :attr:`config.MODEL_ESCALATION_CHAIN`.
    """

    def __init__(
        self,
        escalation_chain: list[str] | None = None,
    ) -> None:
        self._escalation_chain = escalation_chain or list(config.MODEL_ESCALATION_CHAIN)

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, complexity: ComplexityLevel) -> str:
        """
        Return the model ID appropriate for the given complexity level.

        Parameters
        ----------
        complexity: Complexity tier from :class:`ComplexityEstimator`.
        """
        model = _COMPLEXITY_TO_MODEL.get(complexity, config.MEDIUM_MODEL)
        log.info(
            "model_selected",
            complexity=complexity.value,
            model=model,
        )
        return model

    def escalate(self, current_model: str) -> str | None:
        """
        Return the next model up in the escalation chain, or ``None`` if
        the current model is already the strongest available.

        Parameters
        ----------
        current_model: Model ID currently in use.
        """
        chain = self._escalation_chain

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_chain: list[str] = []
        for m in chain:
            if m not in seen:
                unique_chain.append(m)
                seen.add(m)

        try:
            idx = unique_chain.index(current_model)
        except ValueError:
            # current_model is not in the chain — jump to the strongest
            log.warning(
                "model_not_in_chain",
                model=current_model,
                escalating_to=unique_chain[-1],
            )
            return unique_chain[-1]

        if idx + 1 >= len(unique_chain):
            log.warning(
                "model_already_at_max",
                model=current_model,
            )
            return None

        next_model = unique_chain[idx + 1]
        log.info(
            "model_escalated",
            from_model=current_model,
            to_model=next_model,
        )
        return next_model

    @property
    def escalation_chain(self) -> list[str]:
        """The configured escalation chain."""
        return list(self._escalation_chain)
