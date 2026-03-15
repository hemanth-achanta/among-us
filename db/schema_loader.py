"""
Schema loading and formatting for LLM prompt injection.

``SchemaLoader`` accepts a user-supplied schema definition (Python dict or YAML
file) and produces a compact, token-efficient string representation suitable for
inclusion in SQL generation prompts.

Schema definition format
------------------------
The schema is a plain Python dict with the following shape::

    {
        "tables": {
            "orders": {
                "description": "One row per customer order.",
                "columns": {
                    "order_id":    {"type": "INTEGER", "description": "Primary key"},
                    "customer_id": {"type": "INTEGER", "description": "FK to customers"},
                    "created_at":  {"type": "TIMESTAMP", "description": "Order timestamp"},
                },
                "sample_rows": [
                    {"order_id": 1, "customer_id": 42, "created_at": "2024-01-15 10:00:00"},
                ],
            },
            ...
        },
        "relationships": [
            "orders.customer_id → customers.customer_id",
        ]
    }

The schema dict can be supplied programmatically or loaded from a YAML file.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import yaml

from config import config
from utils.logger import get_logger

log = get_logger(__name__)

# Approximate characters-per-token for budget estimation (conservative)
_CHARS_PER_TOKEN = 4


# ─────────────────────────────────────────────────────────────────────────────
# Schema loader
# ─────────────────────────────────────────────────────────────────────────────

class SchemaLoader:
    """
    Loads, validates, and formats database schema metadata for LLM prompts.

    Parameters
    ----------
    schema_def:
        Schema definition dict (see module docstring) **or** a path to a YAML
        file containing the same structure.
    token_limit:
        Approximate token budget for the schema block.  If the formatted schema
        exceeds this limit the loader will truncate sample rows first, then
        column descriptions, to stay within budget.
    max_sample_rows:
        Maximum sample rows to include per table.
    """

    def __init__(
        self,
        schema_def: dict[str, Any] | str | Path,
        token_limit:     int = config.SCHEMA_CONTEXT_TOKEN_LIMIT,
        max_sample_rows: int = config.SCHEMA_SAMPLE_ROWS,
    ) -> None:
        self._token_limit     = token_limit
        self._max_sample_rows = max_sample_rows
        self._schema: dict[str, Any] = self._load(schema_def)
        self._validate()

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(schema_def: dict[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(schema_def, dict):
            return schema_def

        path = Path(schema_def)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)

        if not isinstance(loaded, dict):
            raise ValueError(f"Schema file must contain a YAML mapping, got: {type(loaded)}")

        log.info("schema_loaded_from_file", path=str(path))
        return loaded

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if "tables" not in self._schema:
            raise ValueError("Schema must have a top-level 'tables' key.")
        for table_name, table_def in self._schema["tables"].items():
            if "columns" not in table_def:
                raise ValueError(
                    f"Table '{table_name}' is missing a 'columns' definition."
                )
        log.debug(
            "schema_validated",
            table_count=len(self._schema["tables"]),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def table_names(self) -> list[str]:
        """All table names defined in the schema."""
        return list(self._schema["tables"].keys())

    @property
    def column_names_by_table(self) -> dict[str, list[str]]:
        """Mapping of table → list of column names."""
        return {
            tname: list(tdef["columns"].keys())
            for tname, tdef in self._schema["tables"].items()
        }

    # Layer / subject introspection ------------------------------------------------

    def get_tables_by_layer(self, layer: str) -> list[str]:
        """
        Return all table names that belong to the given logical *layer*
        (e.g. \"raw\", \"semi\", \"processed\").
        """
        target = layer.lower()
        return [
            tname
            for tname, tdef in self._schema["tables"].items()
            if str(tdef.get("layer", "")).lower() == target
        ]

    def get_tables_by_subject(self, subject: str) -> list[str]:
        """
        Return all table names for the given *subject* domain
        (e.g. \"consultations\", \"sessions\").
        """
        target = subject.lower()
        return [
            tname
            for tname, tdef in self._schema["tables"].items()
            if str(tdef.get("subject", "")).lower() == target
        ]

    def get_table_metadata(self, table: str) -> dict[str, Any]:
        """
        Return the metadata dict for *table* including any custom fields
        such as \"layer\", \"subject\", and \"granularity\".
        """
        return dict(self._schema["tables"].get(table, {}))

    def get_all_column_refs(self) -> set[str]:
        """
        Return a set of fully-qualified column references like ``orders.order_id``.
        Used by the SQL validator.
        """
        refs: set[str] = set()
        for tname, tdef in self._schema["tables"].items():
            for col in tdef["columns"]:
                refs.add(f"{tname}.{col}")
                refs.add(col)  # unqualified reference also valid
        return refs

    def format_for_prompt(self) -> str:
        """
        Return a compact schema string for injection into LLM prompts.

        Applies token-budget trimming automatically.
        """
        return self._format_tables_for_prompt(list(self._schema["tables"].keys()))

    def format_summary_for_complexity(self) -> str:
        """
        Return a very short schema summary (just table names + column names)
        for the complexity-estimation prompt where token cost matters most.
        """
        lines: list[str] = ["Tables and columns:"]
        for tname, tdef in self._schema["tables"].items():
            cols = ", ".join(tdef["columns"].keys())
            lines.append(f"  {tname}({cols})")
        return "\n".join(lines)

    def format_layered_summary(self) -> str:
        """
        Return a compact summary of tables grouped by logical layer and subject.
        Includes relationship and join information for the planner.
        """
        by_layer: dict[str, list[str]] = {}
        for tname, tdef in self._schema["tables"].items():
            layer = str(tdef.get("layer", "unspecified")).lower()
            by_layer.setdefault(layer, []).append(tname)

        lines: list[str] = []
        for layer, tables in sorted(by_layer.items()):
            lines.append(f"Layer: {layer}")
            for tname in sorted(tables):
                tdef = self._schema["tables"][tname]
                subject = tdef.get("subject", "unspecified")
                desc = tdef.get("description", "")
                lines.append(f"  - {tname}  (subject: {subject})")
                if desc:
                    lines.append(f"    {textwrap.shorten(desc, width=120, placeholder='…')}")
            lines.append("")

        # Include relationships for the planner
        relationships = self._schema.get("relationships", [])
        if relationships:
            lines.append("Table relationships:")
            for rel in relationships:
                if isinstance(rel, dict):
                    ft = rel.get("from_table", "")
                    fc = rel.get("from_column", "")
                    tt = rel.get("to_table", "")
                    tc = rel.get("to_column", "")
                    desc = rel.get("description", "")
                    lines.append(f"  - {ft}.{fc} -> {tt}.{tc}: {desc}")
                else:
                    lines.append(f"  - {rel}")
            lines.append("")

        join_notes = self._schema.get("join_notes", "")
        if join_notes:
            lines.append("Join guidance:")
            lines.append(join_notes.strip())
        order_cat = self._schema.get("order_category_definition", "")
        if order_cat:
            lines.append("")
            lines.append("Order category definition:")
            lines.append(order_cat.strip())

        return "\n".join(lines).strip()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _format_tables_for_prompt(self, table_names: list[str]) -> str:
        """
        Internal helper to format a subset of tables for prompts.
        Includes relationships and join notes for multi-table context.
        Applies token-budget trimming automatically.
        """
        lines: list[str] = []

        # ── Tables ──────────────────────────────────────────────────────────
        for table_name in table_names:
            table_def = self._schema["tables"].get(table_name)
            if not table_def:
                continue
            description = table_def.get("description", "")
            layer = table_def.get("layer", "")
            subject = table_def.get("subject", "")
            lines.append(f"\n### Table: {table_name}")
            meta_parts = []
            if layer:
                meta_parts.append(f"layer={layer}")
            if subject:
                meta_parts.append(f"subject={subject}")
            if meta_parts:
                lines.append(f"[{', '.join(meta_parts)}]")
            if description:
                lines.append(f"Description: {description}")

            lines.append("Columns:")
            for col_name, col_def in table_def["columns"].items():
                col_type = col_def.get("type", "UNKNOWN")
                col_desc = col_def.get("description", "")
                if col_desc:
                    lines.append(f"  - {col_name} ({col_type}): {col_desc}")
                else:
                    lines.append(f"  - {col_name} ({col_type})")

            sample_rows = table_def.get("sample_rows", [])
            if sample_rows:
                rows_to_show = sample_rows[: self._max_sample_rows]
                lines.append(f"Sample rows (up to {self._max_sample_rows}):")
                for row in rows_to_show:
                    lines.append(f"  {json.dumps(row, default=str)}")

        # ── Relationships (only those involving selected tables) ────────────
        table_set = {t.lower() for t in table_names}
        relationships = self._schema.get("relationships", [])
        relevant_rels = []
        for rel in relationships:
            if isinstance(rel, dict):
                ft = str(rel.get("from_table", "")).lower()
                tt = str(rel.get("to_table", "")).lower()
                if ft in table_set or tt in table_set:
                    relevant_rels.append(rel)
            else:
                relevant_rels.append(rel)

        if relevant_rels:
            lines.append("\n### Relationships")
            for rel in relevant_rels:
                if isinstance(rel, dict):
                    desc = rel.get("description", "")
                    ft = rel.get("from_table", "")
                    fc = rel.get("from_column", "")
                    tt = rel.get("to_table", "")
                    tc = rel.get("to_column", "")
                    rtype = rel.get("type", "")
                    line = f"  - {ft}.{fc} -> {tt}.{tc}"
                    if rtype:
                        line += f" ({rtype})"
                    if desc:
                        line += f": {desc}"
                    lines.append(line)
                else:
                    lines.append(f"  - {rel}")

        # ── Join notes (critical for correct multi-table queries) ───────────
        join_notes = self._schema.get("join_notes", "")
        if join_notes and len(table_names) > 1:
            lines.append(f"\n### Join Notes\n{join_notes.strip()}")
        elif join_notes and len(table_names) == 1:
            # Even for single-table queries, include the table selection guide
            guide_lines = []
            for jline in join_notes.strip().split("\n"):
                stripped = jline.strip()
                if stripped.startswith("4.") or stripped.startswith("- hive."):
                    guide_lines.append(jline)
            if guide_lines:
                lines.append("\n### Table context")
                lines.extend(guide_lines)

        # ── Order category definition (for doc_consult Paid/Free/FreeFollowup) ─
        order_cat = self._schema.get("order_category_definition", "")
        if order_cat:
            lines.append("\n### Order category definition")
            lines.append(order_cat.strip())

        full_text = "\n".join(lines)

        # ── Token-budget enforcement ─────────────────────────────────────────
        char_budget = self._token_limit * _CHARS_PER_TOKEN
        if len(full_text) > char_budget:
            log.warning(
                "schema_truncated",
                original_chars=len(full_text),
                budget_chars=char_budget,
            )
            full_text = self._trim_to_budget(full_text, char_budget)

        return full_text

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _trim_to_budget(text: str, char_budget: int) -> str:
        """
        Naively truncate text to stay within the character budget,
        appending a notice so the LLM knows the schema was truncated.
        """
        notice = "\n\n[Schema truncated to fit context window]"
        available = char_budget - len(notice)
        return text[:available] + notice

    # ── Raw access ────────────────────────────────────────────────────────────

    @property
    def raw(self) -> dict[str, Any]:
        """Return the underlying schema dict."""
        return self._schema
