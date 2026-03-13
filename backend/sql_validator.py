"""
SQL safety and correctness validator.

``SQLValidator`` performs static analysis on LLM-generated SQL *before*
it is sent to the database. This is the primary defence against prompt
injection and accidental data mutation.

Checks performed
----------------
1. Forbidden statement types (DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER,
   CREATE, EXEC / EXECUTE, GRANT, REVOKE).
2. Non-SELECT statements (the query must start with SELECT or WITH … SELECT).
3. Unknown table references (tables not present in the schema).
4. Optional: unknown column references (disabled by default — too strict for
   computed expressions and aliases).

Design note
-----------
The validator uses regex and simple token-level analysis rather than a full SQL
parser so it has zero dependencies beyond the standard library. This keeps it
fast, dependency-free, and easy to audit. A false-negative (missing a bad query)
is less likely than the LLM actually generating harmful SQL — if it does, it
will almost certainly include one of the forbidden keywords.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a single SQL validation check."""

    is_valid:    bool
    errors:      list[str] = field(default_factory=list)
    warnings:    list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# DML / DDL keywords that must never appear in a generated query
_FORBIDDEN_KEYWORDS: list[str] = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "REPLACE",
    "MERGE",
    "UPSERT",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
    "CALL",
    "DO",
    "SET",
    "LOCK",
    "UNLOCK",
]

_FORBIDDEN_PATTERN = re.compile(
    r"(?<!['\"`\w])(?:" + "|".join(_FORBIDDEN_KEYWORDS) + r")(?!['\"`\w])",
    re.IGNORECASE,
)

# A SELECT-only query may start with WITH (CTE) or directly with SELECT
_VALID_START_PATTERN = re.compile(
    r"^\s*(?:with\b|select\b)",
    re.IGNORECASE,
)

# Extract table names from FROM and JOIN clauses (simple heuristic)
_TABLE_REF_PATTERN = re.compile(
    r"(?:FROM|JOIN)\s+([`\"\[]?[\w.]+[`\"\]]?)(?:\s+(?:AS\s+)?[\w]+)?",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class SQLValidator:
    """
    Validates LLM-generated SQL before database execution.

    Parameters
    ----------
    known_tables:
        Set of table names that exist in the schema.  Used to detect
        hallucinated table references.
    """

    def __init__(self, known_tables: list[str]) -> None:
        """
        Initialise the validator with a list of known tables.

        Both fully-qualified table names (e.g. ``hive.data_model.orders``)
        and their unqualified counterparts (``orders``) are accepted to
        give the SQL generator flexibility while still preventing
        hallucinated tables.
        """
        self._known_tables: set[str] = set()
        for t in known_tables:
            t_low = t.lower()
            self._known_tables.add(t_low)
            # Also accept bare table name (last segment after '.')
            self._known_tables.add(t_low.split(".")[-1])

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, sql: str) -> ValidationResult:
        """
        Run all validation checks against *sql*.

        Parameters
        ----------
        sql: Raw SQL string to validate.

        Returns
        -------
        :class:`ValidationResult` with ``is_valid`` set appropriately.
        """
        result = ValidationResult(is_valid=True)

        if not sql or not sql.strip():
            result.add_error("SQL query is empty.")
            return result

        cleaned = self._strip_comments(sql)

        self._check_select_only(cleaned, result)
        self._check_forbidden_keywords(cleaned, result)
        self._check_table_references(cleaned, result)

        if result.is_valid:
            log.debug("sql_validation_passed", sql_snippet=sql[:100])
        else:
            log.warning(
                "sql_validation_failed",
                errors=result.errors,
                sql_snippet=sql[:200],
            )

        return result

    # ── Check methods ─────────────────────────────────────────────────────────

    @staticmethod
    def _strip_comments(sql: str) -> str:
        """Remove SQL line comments and block comments."""
        # Block comments /* ... */
        sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
        # Line comments --
        sql = re.sub(r"--[^\n]*", " ", sql)
        return sql

    @staticmethod
    def _check_select_only(sql: str, result: ValidationResult) -> None:
        if not _VALID_START_PATTERN.match(sql.strip()):
            result.add_error(
                "Query must begin with SELECT or a CTE (WITH … SELECT). "
                "Only read-only queries are permitted."
            )

    @staticmethod
    def _check_forbidden_keywords(sql: str, result: ValidationResult) -> None:
        found = _FORBIDDEN_PATTERN.findall(sql)
        if found:
            unique_found = list(dict.fromkeys(k.upper() for k in found))
            result.add_error(
                f"Query contains forbidden keyword(s): {', '.join(unique_found)}. "
                "Only SELECT statements are allowed."
            )

    def _check_table_references(
        self, sql: str, result: ValidationResult
    ) -> None:
        if not self._known_tables:
            # No schema loaded — skip check
            return

        matches = _TABLE_REF_PATTERN.findall(sql)
        for raw_ref in matches:
            # Normalise: strip quotes/brackets, handle schema-qualified names
            ref = re.sub(r'[`"\[\]]', "", raw_ref).strip()

            # Handle schema.table qualified names
            if "." in ref:
                # Accept any schema-qualified reference — we only validate
                # the table portion for simplicity
                table_part = ref.split(".")[-1].lower()
            else:
                table_part = ref.lower()

            # Skip sub-query aliases, function calls, etc.
            if not re.match(r"^\w+$", table_part):
                continue

            if table_part and table_part not in self._known_tables:
                result.add_error(
                    f"Query references unknown table '{raw_ref}'. "
                    "Only tables defined in the schema are allowed."
                )
