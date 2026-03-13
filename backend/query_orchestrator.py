"""
QueryOrchestrator — the central pipeline coordinator.

This is the only public entry point for the backend. The Streamlit UI calls
:meth:`QueryOrchestrator.run` with a natural-language question and receives a
structured :class:`OrchestratorResult` back.

Pipeline
--------
question
  → ComplexityEstimator         (classify question tier)
  → ModelRouter                 (select initial LLM)
  → RetryManager.reset()        (initialise retry state)
  ↓
  ┌──────────────────────────────────────────────────────┐
  │  Iteration loop (max MAX_QUERY_ITERATIONS)           │
  │                                                      │
  │  SQLGenerator.generate()                             │
  │    ↓ if confidence low → RetryManager.should_retry()│
  │  SQLValidator.validate()                             │
  │    ↓ if invalid      → RetryManager.should_retry()  │
  │  SQLExecutor.execute()                               │
  │    ↓ if error        → RetryManager.should_retry()  │
  │    ↓ if empty        → RetryManager.should_retry()  │
  │  accumulate result                                   │
  │  decide if another iteration is needed               │
  └──────────────────────────────────────────────────────┘
  → ResultInterpreter.interpret()
  → OrchestratorResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from backend.complexity_estimator import ComplexityEstimator
from backend.model_router import ModelRouter
from backend.result_interpreter import ResultInterpreter
from backend.retry_manager import RetryManager, RetryReason
from backend.sql_executor import SQLExecutor, ExecutionResult
from backend.sql_generator import SQLGenerator, SQLGenerationResult
from backend.sql_validator import SQLValidator
from config import config
from config.config import ComplexityLevel
from db.connection_manager import QueryExecutionError
from db.schema_loader import SchemaLoader
from llm.llm_client import LLMClient
from llm.prompt_templates import (
    build_query_planning_messages,
    build_clarification_messages,
)
from utils.logger import Timer, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate step (for UI debug display)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryStep:
    """Record of a single SQL generation + execution cycle."""

    iteration:        int
    model_used:       str
    sql_generated:    str | None
    confidence:       float
    validation_errors: list[str]
    execution_error:  str | None
    rows_returned:    int
    result_summary:   str
    reasoning:        str
    sql_prompt_system:   str
    sql_prompt_messages: list[dict[str, Any]]


@dataclass
class SubQueryPlan:
    """Planned intent for a single SQL query in a multi-query run."""

    id: int
    description: str


@dataclass
class QueryPlan:
    """High-level plan for how many queries to run and why."""

    requires_multiple_queries: bool
    subqueries: list[SubQueryPlan]


# ─────────────────────────────────────────────────────────────────────────────
# Final result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """Structured response returned to the Streamlit UI."""

    answer:            str
    sql_used:          str | None          # SQL from the final (decisive) query
    rows_returned:     int
    model_used:        str                 # Model used for final interpretation
    complexity:        ComplexityLevel
    query_iterations:  int
    steps:             list[QueryStep]     # All intermediate steps
    total_duration_ms: float
    token_summary:     dict[str, int]
    # Prompts used for final answer (for debug display)
    interpretation_prompt_system:   str | None = None
    interpretation_prompt_messages: list[dict[str, Any]] | None = None
    error:             str | None = None   # Set if the pipeline failed completely


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    Coordinates the full analytics pipeline for a single user question.

    Parameters
    ----------
    schema_loader:        Loaded and validated schema metadata.
    connection_manager:   Shared database connection pool.
    llm_client:           Shared Anthropic API client.
    max_iterations:       Max SQL queries per user question.
    max_retries:          Max retry attempts per iteration.
    """

    def __init__(
        self,
        schema_loader:      SchemaLoader,
        connection_manager: Any,   # ConnectionManager | PrestoConnectionManager
        llm_client:         LLMClient,
        max_iterations:     int = config.MAX_QUERY_ITERATIONS,
        max_retries:        int = config.MAX_RETRIES,
    ) -> None:
        self._schema_loader = schema_loader
        self._max_iterations = max_iterations

        # ── Component wiring ─────────────────────────────────────────────────
        self._llm    = llm_client
        self._router = ModelRouter()

        self._schema_summary = schema_loader.format_summary_for_complexity()
        self._complexity_estimator = ComplexityEstimator(
            llm_client=llm_client,
            schema_summary=self._schema_summary,
        )

        dialect = connection_manager.dialect_name
        self._sql_generator = SQLGenerator(
            llm_client=llm_client,
            schema_str=schema_loader.format_for_prompt(),
            dialect=dialect,
        )

        self._sql_validator = SQLValidator(
            known_tables=schema_loader.table_names,
        )

        self._sql_executor = SQLExecutor(
            connection_manager=connection_manager,
        )

        self._result_interpreter = ResultInterpreter(
            llm_client=llm_client,
        )

        self._max_retries = max_retries

        log.info(
            "orchestrator_initialised",
            max_iterations=max_iterations,
            max_retries=max_retries,
            dialect=dialect,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        progress_callback: Optional[Callable[[QueryStep], None]] = None,
    ) -> OrchestratorResult:
        """
        Execute the full analytics pipeline for *question*.

        Parameters
        ----------
        question: Natural-language question from the stakeholder.

        Returns
        -------
        :class:`OrchestratorResult`
        """
        self._llm.reset_usage()
        steps: list[QueryStep] = []

        with Timer() as total_timer:

            # ── 1. Complexity estimation ─────────────────────────────────────
            complexity = self._complexity_estimator.estimate(question)
            log.info("pipeline_start", question=question[:200], complexity=complexity.value)

            # ── 2. Initial model selection ───────────────────────────────────
            initial_model = self._router.select(complexity)

            # ── 3. Retry manager ─────────────────────────────────────────────
            retry_mgr = RetryManager(
                model_router=self._router,
                initial_model=initial_model,
                max_retries=self._max_retries,
            )

            # ── 4. Query planning ─────────────────────────────────────────────
            plan = self._plan_queries(question)

            # ── 5. Iterative query loop following the plan ───────────────────
            previous_queries: list[dict[str, Any]] = []
            final_execution: ExecutionResult | None = None
            final_sql: str | None = None
            final_model: str = initial_model
            iteration = 0
            executed_results: list[ExecutionResult] = []

            subqueries = plan.subqueries or [SubQueryPlan(id=1, description="Answer the question directly.")]
            # Hard cap from config to avoid unbounded multi-query runs
            subqueries = subqueries[: config.MAX_QUERIES_PER_QUESTION]

            for sub_idx, sub in enumerate(subqueries, start=1):
                if iteration >= self._max_iterations:
                    break

                while iteration < self._max_iterations:
                    iteration += 1
                    model = retry_mgr.current_model

                    step, execution, should_continue = self._run_single_iteration(
                        question=question,
                        model=model,
                        iteration=iteration,
                        previous_queries=previous_queries,
                        chat_history=chat_history,
                        retry_mgr=retry_mgr,
                        progress_callback=progress_callback,
                        subquery_description=sub.description,
                    )

                    # Annotate step with planning metadata for debug display
                    step.subquery_id = sub.id
                    step.subquery_description = sub.description
                    steps.append(step)

                    if execution is not None:
                        final_execution = execution
                        final_sql       = step.sql_generated
                        final_model     = model
                        executed_results.append(execution)

                        # Accumulate context for next iteration
                        previous_queries.append({
                            "sql":            step.sql_generated or "",
                            "result_summary": step.result_summary,
                        })

                    if not should_continue:
                        # Move on to the next planned sub-query, if any.
                        break

            # ── 6. Result interpretation ─────────────────────────────────────
            if not executed_results:
                if final_execution is not None and final_execution.row_count == 0:
                    interpretation = self._result_interpreter.interpret_no_results(
                        question=question,
                        sql=final_sql or "",
                        model=final_model,
                        chat_history=chat_history,
                    )
                    rows_returned = 0
                else:
                    # Complete pipeline failure — attempt to ask clarifying questions
                    failure_reasons = "; ".join(
                        [f"step {s.iteration}: {s.execution_error or s.validation_errors}"
                         for s in steps]
                    )
                    clarification_answer, clar_system, clar_messages = (
                        self._generate_clarifying_questions(
                            question=question,
                            failure_reason=failure_reasons,
                            chat_history=chat_history,
                        )
                    )
                    return OrchestratorResult(
                        answer=clarification_answer,
                        sql_used=None,
                        rows_returned=0,
                        model_used=final_model,
                        complexity=complexity,
                        query_iterations=iteration,
                        steps=steps,
                        total_duration_ms=total_timer.elapsed_ms,
                        token_summary=self._token_summary(),
                        interpretation_prompt_system=clar_system,
                        interpretation_prompt_messages=clar_messages,
                        error=failure_reasons,
                    )
            else:
                if len(executed_results) == 1:
                    final_execution = executed_results[-1]
                    interpretation = self._result_interpreter.interpret(
                        question=question,
                        sql=final_execution.sql_executed,
                        dataframe=final_execution.dataframe,
                        model=final_model,
                        chat_history=chat_history,
                    )
                    rows_returned = final_execution.row_count
                    final_sql = final_execution.sql_executed
                else:
                    sql_list = [er.sql_executed for er in executed_results]
                    dataframes = [er.dataframe for er in executed_results]
                    labels = [
                        f"Sub-query {s.subquery_id}: {s.subquery_description}"
                        for s in steps
                        if s.sql_generated
                    ]
                    interpretation = self._result_interpreter.interpret_multi(
                        question=question,
                        sql_list=sql_list,
                        dataframes=dataframes,
                        model=final_model,
                        labels=labels,
                        chat_history=chat_history,
                    )
                    # For summary metrics, report the rows from the last execution.
                    final_execution = executed_results[-1]
                    rows_returned = final_execution.row_count
                    final_sql = final_execution.sql_executed

            final_model = interpretation.model_used

        log.info(
            "pipeline_complete",
            iterations=iteration,
            rows_returned=rows_returned,
            model=final_model,
            total_duration_ms=round(total_timer.elapsed_ms, 1),
        )

        return OrchestratorResult(
            answer=interpretation.answer,
            sql_used=final_sql,
            rows_returned=rows_returned,
            model_used=final_model,
            complexity=complexity,
            query_iterations=iteration,
            steps=steps,
            total_duration_ms=total_timer.elapsed_ms,
            token_summary=self._token_summary(),
            interpretation_prompt_system=interpretation.prompt_system,
            interpretation_prompt_messages=interpretation.prompt_messages,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _plan_queries(self, question: str) -> QueryPlan:
        """
        Use the LLM to decide whether multiple queries are needed and, if so,
        return a small list of sub-query intents.

        Falls back to a single-query plan if planning fails or returns
        inconsistent data.
        """
        import json

        messages, system = build_query_planning_messages(
            question=question,
            schema_summary=self._schema_summary,
            max_queries=config.MAX_QUERIES_PER_QUESTION,
        )

        try:
            with Timer() as t:
                response = self._llm.complete(
                    model=config.LOW_MODEL,
                    messages=messages,
                    system=system,
                    max_tokens=256,
                )

            raw = response.content
            parsed = json.loads(raw)
            requires_multi = bool(parsed.get("requires_multiple_queries", False))
            queries = parsed.get("queries") or []

            subqueries: list[SubQueryPlan] = []
            for idx, q in enumerate(queries, start=1):
                desc = str(q.get("description", "")).strip()
                if not desc:
                    continue
                subqueries.append(SubQueryPlan(id=idx, description=desc))

            if not subqueries:
                requires_multi = False

            log.info(
                "query_planning_complete",
                requires_multiple=requires_multi,
                planned_queries=len(subqueries),
                duration_ms=round(t.elapsed_ms, 1),
            )

            return QueryPlan(
                requires_multiple_queries=requires_multi,
                subqueries=subqueries,
            )

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "query_planning_failed",
                error=str(exc),
            )
            # Conservative fallback: single-query plan
            return QueryPlan(
                requires_multiple_queries=False,
                subqueries=[SubQueryPlan(id=1, description="Answer the question directly.")],
            )

    def _generate_clarifying_questions(
        self,
        question: str,
        failure_reason: str,
        chat_history: list[dict[str, str]] | None,
    ) -> tuple[str, str | None, list[dict[str, Any]] | None]:
        """
        Ask the LLM to produce one or two clarifying questions for the user
        when the pipeline cannot safely generate SQL.

        Returns
        -------
        (answer_text, system_prompt_used, messages_used)
        """
        messages, system = build_clarification_messages(
            question=question,
            schema_summary=self._schema_summary,
            failure_reason=failure_reason,
        )

        # Optionally fold recent chat history into the first user message.
        # For simplicity we rely on the existing schema summary + failure reason
        # and do not re-thread the entire conversation here.
        try:
            with Timer() as t:
                response = self._llm.complete(
                    model=config.LOW_MODEL,
                    messages=messages,
                    system=system,
                    max_tokens=256,
                )

            log.info(
                "clarification_questions_generated",
                duration_ms=round(t.elapsed_ms, 1),
            )

            return response.content.strip(), system, messages

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "clarification_generation_failed",
                error=str(exc),
            )
            fallback = (
                "I wasn't able to generate a safe SQL query for your question. "
                "Could you clarify what time range, key metrics, and any important "
                "filters (such as product, region, or status) you are interested in?"
            )
            return fallback, None, None

    def _run_single_iteration(
        self,
        question: str,
        model: str,
        iteration: int,
        previous_queries: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None,
        retry_mgr: RetryManager,
        progress_callback: Optional[Callable[[QueryStep], None]] = None,
        subquery_description: str | None = None,
    ) -> tuple[QueryStep, ExecutionResult | None, bool]:
        """
        Run one SQL generation → validation → execution cycle.

        Returns
        -------
        (step, execution_result, should_continue_iterating)
        """
        gen_result: SQLGenerationResult | None = None
        exec_result: ExecutionResult | None = None
        validation_errors: list[str] = []
        execution_error: str | None = None
        result_summary = ""
        should_continue = False

        # ── a. SQL generation ────────────────────────────────────────────────
        try:
            gen_result = self._sql_generator.generate(
                question=question,
                model=model,
                previous_queries=previous_queries,
                chat_history=chat_history,
                subquery_description=subquery_description,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("sql_generation_exception", error=str(exc), iteration=iteration)
            step = QueryStep(
                iteration=iteration, model_used=model,
                sql_generated=None, confidence=0.0,
                validation_errors=[], execution_error=str(exc),
                rows_returned=0, result_summary="", reasoning="",
                sql_prompt_system="", sql_prompt_messages=[],
            )
            retry_mgr.should_retry(RetryReason.EXECUTION_ERROR, str(exc))
            if progress_callback is not None:
                try:
                    progress_callback(step)
                except Exception as cb_exc:  # noqa: BLE001
                    log.warning("progress_callback_error", error=str(cb_exc))
            return step, None, False

        if gen_result.sql_query is None:
            step = QueryStep(
                iteration=iteration, model_used=model,
                sql_generated=None, confidence=gen_result.confidence,
                validation_errors=[],
                execution_error="LLM could not generate SQL for this question.",
                rows_returned=0, result_summary="",
                reasoning=gen_result.reasoning,
                sql_prompt_system=gen_result.prompt_system,
                sql_prompt_messages=gen_result.prompt_messages,
            )
            if progress_callback is not None:
                try:
                    progress_callback(step)
                except Exception as cb_exc:  # noqa: BLE001
                    log.warning("progress_callback_error", error=str(cb_exc))
            return step, None, False

        # ── b. Confidence check ──────────────────────────────────────────────
        if retry_mgr.needs_retry_for_confidence(gen_result.confidence):
            log.info(
                "low_confidence_retry",
                confidence=gen_result.confidence,
                model=model,
            )
            if retry_mgr.should_retry(
                RetryReason.LOW_CONFIDENCE,
                f"confidence={gen_result.confidence:.2f}",
            ):
                # Re-run with (possibly escalated) model
                try:
                    gen_result = self._sql_generator.generate(
                        question=question,
                        model=retry_mgr.current_model,
                        previous_queries=previous_queries,
                    )
                    model = retry_mgr.current_model
                except Exception as exc:  # noqa: BLE001
                    log.error("sql_regen_exception", error=str(exc))

        # ── c. Validation ────────────────────────────────────────────────────
        val_result = self._sql_validator.validate(gen_result.sql_query)
        if not val_result.is_valid:
            validation_errors = val_result.errors
            if retry_mgr.should_retry(
                RetryReason.VALIDATION_ERROR,
                "; ".join(validation_errors),
            ):
                # Retry with next model — will be picked up in next outer loop
                step = QueryStep(
                    iteration=iteration, model_used=model,
                    sql_generated=gen_result.sql_query,
                    confidence=gen_result.confidence,
                    validation_errors=validation_errors,
                    execution_error=None, rows_returned=0,
                    result_summary="", reasoning=gen_result.reasoning,
                    sql_prompt_system=gen_result.prompt_system,
                    sql_prompt_messages=gen_result.prompt_messages,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(step)
                    except Exception as cb_exc:  # noqa: BLE001
                        log.warning("progress_callback_error", error=str(cb_exc))
                return step, None, True  # continue outer loop
            else:
                step = QueryStep(
                    iteration=iteration, model_used=model,
                    sql_generated=gen_result.sql_query,
                    confidence=gen_result.confidence,
                    validation_errors=validation_errors,
                    execution_error="Validation failed and retries exhausted.",
                    rows_returned=0, result_summary="",
                    reasoning=gen_result.reasoning,
                    sql_prompt_system=gen_result.prompt_system,
                    sql_prompt_messages=gen_result.prompt_messages,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(step)
                    except Exception as cb_exc:  # noqa: BLE001
                        log.warning("progress_callback_error", error=str(cb_exc))
                return step, None, False

        # ── d. Execution ─────────────────────────────────────────────────────
        try:
            exec_result = self._sql_executor.execute(gen_result.sql_query)
        except QueryExecutionError as exc:
            execution_error = str(exc)
            if retry_mgr.should_retry(RetryReason.EXECUTION_ERROR, execution_error):
                step = QueryStep(
                    iteration=iteration, model_used=model,
                    sql_generated=gen_result.sql_query,
                    confidence=gen_result.confidence,
                    validation_errors=[], execution_error=execution_error,
                    rows_returned=0, result_summary="",
                    reasoning=gen_result.reasoning,
                    sql_prompt_system=gen_result.prompt_system,
                    sql_prompt_messages=gen_result.prompt_messages,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(step)
                    except Exception as cb_exc:  # noqa: BLE001
                        log.warning("progress_callback_error", error=str(cb_exc))
                return step, None, True  # continue outer loop
            else:
                step = QueryStep(
                    iteration=iteration, model_used=model,
                    sql_generated=gen_result.sql_query,
                    confidence=gen_result.confidence,
                    validation_errors=[], execution_error=execution_error,
                    rows_returned=0, result_summary="",
                    reasoning=gen_result.reasoning,
                    sql_prompt_system=gen_result.prompt_system,
                    sql_prompt_messages=gen_result.prompt_messages,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(step)
                    except Exception as cb_exc:  # noqa: BLE001
                        log.warning("progress_callback_error", error=str(cb_exc))
                return step, None, False

        # ── e. Empty result check ────────────────────────────────────────────
        if exec_result.row_count == 0:
            if retry_mgr.should_retry(
                RetryReason.EMPTY_RESULT,
                "query returned 0 rows",
            ):
                result_summary = "Query returned 0 rows."
                step = QueryStep(
                    iteration=iteration, model_used=model,
                    sql_generated=gen_result.sql_query,
                    confidence=gen_result.confidence,
                    validation_errors=[], execution_error=None,
                    rows_returned=0, result_summary=result_summary,
                    reasoning=gen_result.reasoning,
                    sql_prompt_system=gen_result.prompt_system,
                    sql_prompt_messages=gen_result.prompt_messages,
                )
                previous_queries.append({
                    "sql": gen_result.sql_query,
                    "result_summary": result_summary,
                })
                if progress_callback is not None:
                    try:
                        progress_callback(step)
                    except Exception as cb_exc:  # noqa: BLE001
                        log.warning("progress_callback_error", error=str(cb_exc))
                return step, exec_result, True  # try another angle

        # ── f. Successful result ──────────────────────────────────────────────
        result_summary = self._sql_executor.summarise_result(exec_result)

        # Decide if another iteration is worthwhile
        # (high complexity questions may benefit from follow-up queries)
        should_continue = False  # conservative default; orchestrator manages iteration count

        step = QueryStep(
            iteration=iteration, model_used=model,
            sql_generated=gen_result.sql_query,
            confidence=gen_result.confidence,
            validation_errors=[], execution_error=None,
            rows_returned=exec_result.row_count,
            result_summary=result_summary,
            reasoning=gen_result.reasoning,
            sql_prompt_system=gen_result.prompt_system,
            sql_prompt_messages=gen_result.prompt_messages,
        )

        if progress_callback is not None:
            try:
                progress_callback(step)
            except Exception as cb_exc:  # noqa: BLE001
                log.warning("progress_callback_error", error=str(cb_exc))

        return step, exec_result, should_continue

    def _token_summary(self) -> dict[str, int]:
        usage = self._llm.usage
        return {
            "input_tokens":  usage.total_input,
            "output_tokens": usage.total_output,
            "total_tokens":  usage.total_tokens,
            "llm_calls":     usage.call_count,
        }
