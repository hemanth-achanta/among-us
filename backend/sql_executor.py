"""
SQL execution layer.

Supports two connection backends transparently:

``PrestoConnectionManager``  (Trino / Presto)
    Uses ``prestodb`` directly via ``PrestoConnectionManager.execute_query()``.
    Read-only by nature — Trino SELECT-only workloads, no commit needed.

``ConnectionManager``  (PostgreSQL / MySQL / Snowflake via SQLAlchemy)
    Wraps every query in a transaction that is *always rolled back*, ensuring
    nothing is committed even if the validator passes a mutation through.

Both paths return an identical :class:`ExecutionResult` so the orchestrator
and UI never need to know which backend is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import pandas as pd

from config import config
from utils.logger import Timer, get_logger

if TYPE_CHECKING:
    from db.connection_manager import ConnectionManager, PrestoConnectionManager

log = get_logger(__name__)

# Type alias for either manager
AnyConnectionManager = Union["ConnectionManager", "PrestoConnectionManager"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Outcome of a single SQL execution."""

    dataframe:    pd.DataFrame
    row_count:    int
    column_names: list[str]
    duration_ms:  float
    sql_executed: str
    truncated:    bool   # True when rows were capped at MAX_RESULT_ROWS


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class SQLExecutor:
    """
    Executes validated SQL queries via the configured connection manager.

    Automatically detects whether a ``PrestoConnectionManager`` (Trino) or a
    ``ConnectionManager`` (SQLAlchemy) has been supplied and routes accordingly.

    Parameters
    ----------
    connection_manager: Either a :class:`PrestoConnectionManager` or a
                        :class:`ConnectionManager`.
    max_rows:           Row cap (defaults from config).
    """

    def __init__(
        self,
        connection_manager: AnyConnectionManager,
        max_rows: int = config.MAX_RESULT_ROWS,
    ) -> None:
        self._cm       = connection_manager
        self._max_rows = max_rows
        self._is_presto = self._detect_presto(connection_manager)

        log.info(
            "sql_executor_init",
            backend="presto" if self._is_presto else "sqlalchemy",
            max_rows=max_rows,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, sql: str) -> ExecutionResult:
        """
        Execute *sql* and return an :class:`ExecutionResult`.

        For SQLAlchemy connections the query runs inside a read-only
        transaction (never committed).  For Presto/Trino the connection is
        inherently read-only at the cursor level.

        Parameters
        ----------
        sql: Validated SQL string to execute.

        Raises
        ------
        QueryExecutionError: If the database reports an error.
        """
        from db.connection_manager import QueryExecutionError

        log.info("sql_execution_start", sql_snippet=sql[:200])

        with Timer() as t:
            try:
                if self._is_presto:
                    df, truncated = self._execute_presto(sql)
                else:
                    df, truncated = self._execute_sqlalchemy(sql)
            except QueryExecutionError:
                raise
            except Exception as exc:
                log.error(
                    "sql_execution_failed",
                    error=str(exc),
                    sql_snippet=sql[:200],
                    duration_ms=round(t.elapsed_ms, 1),
                )
                raise QueryExecutionError(str(exc)) from exc

        result = ExecutionResult(
            dataframe=df,
            row_count=len(df),
            column_names=list(df.columns),
            duration_ms=t.elapsed_ms,
            sql_executed=sql,
            truncated=truncated,
        )

        log.info(
            "sql_execution_complete",
            row_count=result.row_count,
            column_count=len(result.column_names),
            truncated=truncated,
            duration_ms=round(t.elapsed_ms, 1),
        )

        return result

    # ── Backend-specific execution ────────────────────────────────────────────

    def _execute_presto(self, sql: str) -> tuple[pd.DataFrame, bool]:
        """
        Execute via PrestoConnectionManager (Trino / prestodb).

        Delegates entirely to ``PrestoConnectionManager.execute_query()`` which
        mirrors the notebook's ``df_from_presto()`` pattern.
        """
        return self._cm.execute_query(sql, max_rows=self._max_rows)  # type: ignore[union-attr]

    def _execute_sqlalchemy(self, sql: str) -> tuple[pd.DataFrame, bool]:
        """
        Execute via SQLAlchemy ConnectionManager in a read-only transaction.
        """
        import sqlalchemy as sa

        with self._cm.get_connection() as conn:  # type: ignore[union-attr]
            with conn.begin():
                cursor_result = conn.execute(sa.text(sql))
                rows          = cursor_result.fetchmany(self._max_rows + 1)
                col_names     = list(cursor_result.keys())
            conn.rollback()   # belt-and-suspenders: never commit

        truncated    = len(rows) > self._max_rows
        limited_rows = rows[: self._max_rows]
        df           = pd.DataFrame(limited_rows, columns=col_names)
        return df, truncated

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_presto(manager: AnyConnectionManager) -> bool:
        """Return True if *manager* is a PrestoConnectionManager."""
        return hasattr(manager, "execute_query")

    def summarise_result(
        self,
        result: ExecutionResult,
        max_chars: int = 1_000,
    ) -> str:
        """
        Return a compact string summary of a query result for prompt injection.

        Used when folding previous-iteration results into follow-up prompts.
        """
        if result.row_count == 0:
            return "Query returned 0 rows."

        try:
            table_str = result.dataframe.head(10).to_string(index=False)
            if result.truncated:
                table_str += f"\n... (result truncated to {self._max_rows} rows)"
        except Exception:  # noqa: BLE001
            cols      = ", ".join(result.column_names)
            table_str = f"Columns: {cols} — {result.row_count} rows total"

        summary = f"Rows returned: {result.row_count}\n{table_str}"

        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n... [summary truncated]"

        return summary
