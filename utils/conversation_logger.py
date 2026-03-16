"""
Per-conversation trace logger.

Captures the COMPLETE pipeline trace for every question-answer cycle into a
single, self-contained JSON file.  Each file includes:

  - The user question and chat history sent
  - Complexity estimation details
  - Query plan
  - Every SQL generation step (full prompts, raw LLM responses, stop reasons)
  - Validation and execution results
  - Result interpretation (full prompts, raw LLM response, data sizes)
  - Truncation events (where data was cut short, with before/after sizes)
  - Final answer, token usage, timing

This makes it possible to open any conversation log and immediately see
where something went wrong (e.g. LLM output truncated, table data too large
for the prompt, etc.).

Usage (from QueryOrchestrator)::

    clog = ConversationLogger(question)
    clog.log_stage("complexity_estimation", ...)
    ...
    clog.finalize(result)
    # → writes logs/conversations/<conversation_id>.json
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from config import config

log = get_logger(__name__)


def _delete_other_conversation_logs(log_dir: Path, keep: str) -> None:
    """Delete all JSON files in log_dir except the file named ``keep`` (last conversation only)."""
    if not log_dir.is_dir():
        return
    removed = 0
    for path in log_dir.glob("*.json"):
        if path.name != keep:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                log.warning("conversation_log_delete_failed", path=str(path), error=str(exc))
    if removed:
        log.debug("conversation_logs_pruned", kept=keep, removed_count=removed)


class ConversationLogger:
    """Accumulates pipeline events and writes a full JSON trace on finalize."""

    def __init__(
        self,
        question: str,
        log_dir: str | Path = "logs/conversations",
        *,
        chat_history: list[dict[str, str]] | None = None,
    ) -> None:
        self.conversation_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.question = question
        self.chat_history_sent = chat_history or []
        self._log_dir = Path(log_dir)
        self._stages: list[dict[str, Any]] = []
        self._truncation_events: list[dict[str, Any]] = []
        self._started_at = datetime.now(timezone.utc)
        self._finalized = False
        self._step_index: int = 0

    def log_stage(self, stage: str, **data: Any) -> None:
        """Append a pipeline stage event."""
        self._step_index += 1
        entry: dict[str, Any] = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_index": self._step_index,
            **data,
        }
        self._stages.append(entry)

    def log_truncation(
        self,
        location: str,
        original_size: int,
        truncated_size: int,
        unit: str = "chars",
        detail: str = "",
    ) -> None:
        """Record a data truncation event for later inspection."""
        event = {
            "location": location,
            "original_size": original_size,
            "truncated_size": truncated_size,
            "unit": unit,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._truncation_events.append(event)
        log.info(
            "truncation_detected",
            location=location,
            original_size=original_size,
            truncated_size=truncated_size,
            unit=unit,
        )

    def log_llm_call(
        self,
        purpose: str,
        model: str,
        prompt_system: str,
        prompt_messages: list[dict[str, Any]],
        raw_response: str,
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        max_tokens_budget: int | None = None,
        **extra: Any,
    ) -> None:
        """
        Record a complete LLM call with all inputs and outputs.

        ``stop_reason`` is critical: "end_turn" means the LLM finished
        naturally; "max_tokens" means the output was truncated.
        """
        was_truncated = stop_reason == "max_tokens"
        self._step_index += 1

        # Estimate per-call API cost using the shared MODEL_PRICING config.
        pricing: dict[str, tuple[float, float]] = getattr(config, "MODEL_PRICING", {})
        default_per_million: tuple[float, float] = (3.0, 15.0)
        in_per_m, out_per_m = pricing.get(model, default_per_million)
        cost_input_usd = (input_tokens / 1_000_000.0) * in_per_m
        cost_output_usd = (output_tokens / 1_000_000.0) * out_per_m
        total_cost_usd = round(cost_input_usd + cost_output_usd, 8)

        # Optionally express cost in configured display currency.
        display_currency = getattr(config, "COST_DISPLAY_CURRENCY", "USD")
        usd_to_inr = getattr(config, "USD_TO_INR", 92.0)
        if display_currency == "INR":
            total_cost_display = round(total_cost_usd * usd_to_inr, 6)
        else:
            total_cost_display = total_cost_usd
        entry: dict[str, Any] = {
            "stage": f"llm_call__{purpose}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_index": self._step_index,
            "model": model,
            "prompt_system": prompt_system,
            "prompt_system_length": len(prompt_system),
            "prompt_messages": prompt_messages,
            "prompt_messages_total_chars": sum(
                len(m.get("content", "")) for m in prompt_messages
            ),
            "raw_response": raw_response,
            "raw_response_length": len(raw_response),
            "stop_reason": stop_reason,
            "output_was_truncated": was_truncated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": total_cost_usd,
            "estimated_cost_display": total_cost_display,
            "cost_display_currency": display_currency,
            "max_tokens_budget": max_tokens_budget,
            "duration_ms": round(duration_ms, 1),
            **extra,
        }
        self._stages.append(entry)

        if was_truncated:
            self.log_truncation(
                location=f"llm_output__{purpose}",
                original_size=output_tokens,
                truncated_size=output_tokens,
                unit="tokens",
                detail=(
                    f"LLM output hit max_tokens limit "
                    f"(budget={max_tokens_budget}, used={output_tokens}). "
                    "The response was cut off mid-stream."
                ),
            )

    def log_data_snapshot(
        self,
        label: str,
        *,
        row_count: int | None = None,
        column_count: int | None = None,
        column_names: list[str] | None = None,
        data_chars: int | None = None,
        preview: str = "",
    ) -> None:
        """Record the shape and a preview of a DataFrame at a pipeline stage."""
        self._step_index += 1
        self._stages.append({
            "stage": f"data_snapshot__{label}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_index": self._step_index,
            "row_count": row_count,
            "column_count": column_count,
            "column_names": column_names,
            "data_chars": data_chars,
            "preview": preview[:2000] if preview else "",
        })

    def finalize(
        self,
        *,
        answer: str = "",
        sql_used: str | None = None,
        rows_returned: int = 0,
        model_used: str = "",
        complexity: str = "",
        query_iterations: int = 0,
        total_duration_ms: float = 0.0,
        token_summary: dict[str, int | float] | None = None,
        error: str | None = None,
    ) -> str:
        """
        Write the complete conversation trace to a JSON file.

        Returns the file path (so the UI can reference it).
        """
        if self._finalized:
            return str(self._log_dir / f"{self.conversation_id}.json")

        # Full session chat: prior turns + current user question + current assistant answer (with meta)
        assistant_meta: dict[str, Any] = {
            "sql_used": sql_used,
            "rows_returned": rows_returned,
            "model_used": model_used,
            "complexity": complexity,
            "query_iterations": query_iterations,
            "total_duration_ms": round(total_duration_ms, 1),
            "token_summary": token_summary or {},
            "error": error,
        }
        session_chat: list[dict[str, Any]] = [
            *self.chat_history_sent,
            {"role": "user", "content": self.question},
            {
                "role": "assistant",
                "content": answer,
                "meta": assistant_meta,
            },
        ]

        trace: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "question": self.question,
            "chat_history_sent": self.chat_history_sent,
            "session_chat": session_chat,
            "final_result": {
                "answer": answer,
                "answer_length": len(answer),
                "sql_used": sql_used,
                "rows_returned": rows_returned,
                "model_used": model_used,
                "complexity": complexity,
                "query_iterations": query_iterations,
                "total_duration_ms": round(total_duration_ms, 1),
                "token_summary": token_summary or {},
                "error": error,
            },
            "truncation_events": self._truncation_events,
            "truncation_count": len(self._truncation_events),
            "pipeline_stages": self._stages,
            "stage_count": len(self._stages),
        }

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._log_dir / f"{self.conversation_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, default=str, ensure_ascii=False)

            self._finalized = True
            # Keep only this conversation log; delete all others (last conversation only)
            _delete_other_conversation_logs(self._log_dir, keep=file_path.name)
            log.info(
                "conversation_trace_written",
                conversation_id=self.conversation_id,
                file_path=str(file_path),
                stages=len(self._stages),
                truncations=len(self._truncation_events),
            )
            return str(file_path)

        except Exception as exc:
            log.error(
                "conversation_trace_write_failed",
                conversation_id=self.conversation_id,
                error=str(exc),
            )
            return ""
