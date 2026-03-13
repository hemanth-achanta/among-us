"""
All LLM prompt templates used by the analytics assistant.

Design principles
-----------------
* Templates are pure strings with named `{placeholders}`.
* Each template is a module-level constant so it can be reviewed / audited
  without executing any logic.
* Helper functions (`build_*`) assemble context and return a final string
  ready to be sent to the LLM.
"""
from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# SQL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

SQL_GENERATION_SYSTEM = """\
You are an expert SQL analyst assistant.
Your sole job is to translate natural-language questions into correct, safe SQL queries.

Rules you MUST follow (non-negotiable):
1. Only reference tables and columns that exist in the schema provided below.
2. NEVER fabricate table names, column names, or values.
3. Only generate SELECT statements. Never produce DROP, DELETE, UPDATE, INSERT,
   TRUNCATE, ALTER, CREATE, or any DDL/DML statement.
4. Always qualify column names with their table name or alias when a JOIN is present.
5. Use standard SQL unless the dialect specified below requires otherwise.
6. Limit result sets to a maximum of {max_rows} rows using LIMIT / TOP / FETCH FIRST
   appropriate for the dialect.
7. If the question cannot be answered with the available schema, set
   "sql_query" to null and explain why in "reasoning".

Output format (respond with ONLY this JSON, no markdown fences):
{{
  "sql_query": "<valid SQL string or null>",
  "reasoning": "<brief explanation of your approach>",
  "confidence": <float between 0.0 and 1.0>
}}
"""

SQL_GENERATION_USER = """\
## Database dialect
{dialect}

## Schema
{schema}

{conversation_history}

## User question
{question}

{subquery_intent}
{previous_context}
"""

_PREVIOUS_CONTEXT_BLOCK = """\
## Previous query results (for context)
The following queries were already executed in this session.
Use their results to inform your next query; do not repeat them.

{previous_queries}
"""

_CONVERSATION_HISTORY_BLOCK = """\
## Conversation history
The following prior messages are from this chat session (oldest first):

{history}
"""


# ─────────────────────────────────────────────────────────────────────────────
# RESULT INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

RESULT_INTERPRETATION_SYSTEM = """\
You are a concise data analyst communicating query results to business stakeholders.

Rules:
1. Answer the user's original question directly using the data provided.
2. Reference specific numbers, percentages, and row values from the query output.
3. Do NOT speculate beyond what the data shows.
4. Do NOT suggest further analysis unless the user explicitly asked for it.
5. Keep the response to 3-5 sentences maximum unless more detail is required.
6. Use plain business language — avoid SQL or technical jargon.
"""

RESULT_INTERPRETATION_USER = """\
{conversation_history}

## Original question
{question}

## SQL used
{sql}

## Query results ({row_count} rows)
{results_table}

Please answer the original question based on the data above.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_ESTIMATION_SYSTEM = """\
You are a SQL complexity classifier.
Given a natural-language analytics question and a database schema, classify the
question into one of three complexity levels:

  low    – simple aggregation, single table, basic filter
  medium – joins, grouped analysis, time-based filtering, simple sub-queries
  high   – multi-step reasoning, multiple joins, ambiguous intent, iterative
            exploration required, cross-domain analysis

Output ONLY this JSON (no markdown):
{{
  "complexity": "<low|medium|high>",
  "reasoning": "<one-sentence justification>"
}}
"""

COMPLEXITY_ESTIMATION_USER = """\
## Schema summary
{schema_summary}

## Question
{question}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper builders
# ─────────────────────────────────────────────────────────────────────────────

def build_sql_generation_messages(
    question: str,
    schema: str,
    dialect: str,
    max_rows: int,
    previous_queries: list[dict[str, Any]] | None = None,
    chat_history: list[dict[str, str]] | None = None,
    subquery_description: str | None = None,
) -> list[dict[str, str]]:
    """
    Assemble the messages list for a SQL generation call.

    Parameters
    ----------
    question:         Natural-language question from the user.
    schema:           Formatted schema string (from SchemaLoader).
    dialect:          SQL dialect name, e.g. "PostgreSQL", "MySQL", "Snowflake".
    max_rows:         Row limit to inject into the system prompt.
    previous_queries: Optional list of dicts with keys "sql" and "result_summary".

    Returns
    -------
    List of {"role": ..., "content": ...} dicts ready for the Anthropic API.
    """
    system = SQL_GENERATION_SYSTEM.format(max_rows=max_rows)

    previous_context = ""
    if previous_queries:
        lines = []
        for i, pq in enumerate(previous_queries, start=1):
            lines.append(f"### Query {i}\nSQL: {pq['sql']}\nResult: {pq['result_summary']}")
        previous_context = _PREVIOUS_CONTEXT_BLOCK.format(
            previous_queries="\n\n".join(lines)
        )

    conversation_history = ""
    if chat_history:
        hist_lines: list[str] = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if not content:
                continue
            hist_lines.append(f"{role}: {content}")
        if hist_lines:
            conversation_history = _CONVERSATION_HISTORY_BLOCK.format(
                history="\n".join(hist_lines)
            )

    subquery_intent_block = ""
    if subquery_description:
        subquery_intent_block = (
            "## Current sub-query intent\n"
            f"{subquery_description}\n"
        )

    user_content = SQL_GENERATION_USER.format(
        dialect=dialect,
        schema=schema,
        question=question,
        conversation_history=conversation_history,
        subquery_intent=subquery_intent_block,
        previous_context=previous_context,
    )

    return [
        {"role": "user", "content": user_content},
    ], system


def build_interpretation_messages(
    question: str,
    sql: str,
    results_table: str,
    row_count: int,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble messages for result interpretation.

    Returns
    -------
    (messages, system_prompt) tuple.
    """
    conversation_history = ""
    if chat_history:
        hist_lines: list[str] = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if not content:
                continue
            hist_lines.append(f"{role}: {content}")
        if hist_lines:
            conversation_history = _CONVERSATION_HISTORY_BLOCK.format(
                history="\n".join(hist_lines)
            )

    user_content = RESULT_INTERPRETATION_USER.format(
        question=question,
        sql=sql,
        results_table=results_table,
        row_count=row_count,
        conversation_history=conversation_history,
    )
    return [{"role": "user", "content": user_content}], RESULT_INTERPRETATION_SYSTEM


def build_complexity_messages(
    question: str,
    schema_summary: str,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble messages for complexity estimation.

    Returns
    -------
    (messages, system_prompt) tuple.
    """
    user_content = COMPLEXITY_ESTIMATION_USER.format(
        schema_summary=schema_summary,
        question=question,
    )
    return [{"role": "user", "content": user_content}], COMPLEXITY_ESTIMATION_SYSTEM


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PLANNING
# ─────────────────────────────────────────────────────────────────────────────

QUERY_PLANNING_SYSTEM = """\
You are an analytics planning assistant.
Given a natural-language question and a database schema summary, decide whether
the question can be answered with a single SQL query or if multiple distinct
queries are required.

Rules:
1. Prefer a single query when it can answer the question clearly.
2. Only propose multiple queries when they each answer a clearly different
   sub-part of the question (for example, overall totals vs. detailed breakdowns).
3. You are NOT writing SQL here, only planning query intents in plain English.
4. At most {max_queries} planned queries are allowed.

Output ONLY this JSON (no markdown fences):
{{
  "requires_multiple_queries": <true|false>,
  "queries": [
    {{
      "id": 1,
      "description": "Short description of what this query should compute"
    }}
  ]
}}
"""

QUERY_PLANNING_USER = """\
## Schema summary
{schema_summary}

## User question
{question}
"""


def build_query_planning_messages(
    question: str,
    schema_summary: str,
    max_queries: int,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble messages for the query-planning step.

    Returns
    -------
    (messages, system_prompt) tuple.
    """
    system = QUERY_PLANNING_SYSTEM.format(max_queries=max_queries)
    user_content = QUERY_PLANNING_USER.format(
        schema_summary=schema_summary,
        question=question,
    )
    return [{"role": "user", "content": user_content}], system


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RESULT INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_interpretation_messages(
    question: str,
    queries: list[dict[str, str]],
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble messages for interpreting multiple query results at once.

    Each entry in *queries* should have:
      - "label": human-friendly label (e.g. "Query 1 – totals")
      - "sql": SQL string that was executed
      - "results_table": formatted results table
      - "row_count": number of rows in the result
    """
    conversation_history = ""
    if chat_history:
        hist_lines: list[str] = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if not content:
                continue
            hist_lines.append(f"{role}: {content}")
        if hist_lines:
            conversation_history = _CONVERSATION_HISTORY_BLOCK.format(
                history="\n".join(hist_lines)
            )

    parts: list[str] = []
    parts.append(conversation_history)
    parts.append("## Original question")
    parts.append(question)
    parts.append("")
    parts.append("## Query results")

    for q in queries:
        label = q.get("label", "Query")
        sql = q.get("sql", "")
        results_table = q.get("results_table", "")
        row_count = q.get("row_count", 0)
        parts.append(f"\n### {label}")
        parts.append("SQL:")
        parts.append(sql)
        parts.append("")
        parts.append(f"Results ({row_count} rows)")
        parts.append(results_table)
        parts.append("")

    parts.append(
        "Please answer the original question based on all of the data above."
    )

    user_content = "\n".join(parts)
    return [{"role": "user", "content": user_content}], RESULT_INTERPRETATION_SYSTEM


# ─────────────────────────────────────────────────────────────────────────────
# CLARIFICATION (FOLLOW-UP QUESTIONS)
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_SYSTEM = """\
You are a helpful analytics assistant.

When the current database schema is insufficient or too ambiguous to safely
generate a SQL query, your job is to ask the user one or two short, specific
clarifying questions instead of attempting to answer directly.

Rules:
1. Do NOT attempt to write SQL.
2. Ask at most two concise questions.
3. Focus on details that would most help transform the question into a
   concrete, answerable analytics query (e.g. time range, metrics, filters,
   table choice, or ambiguity in definitions).
4. Speak directly to the user in plain language.
"""

CLARIFICATION_USER = """\
The assistant was unable to safely generate SQL for the following question:

Original question:
{question}

Schema summary:
{schema_summary}

Internal failure reason:
{failure_reason}

Please respond with one or two short clarifying questions for the user that
would help you answer this question with SQL.
"""


def build_clarification_messages(
    question: str,
    schema_summary: str,
    failure_reason: str,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble messages for asking the LLM to generate clarifying questions.

    Returns
    -------
    (messages, system_prompt) tuple.
    """
    user_content = CLARIFICATION_USER.format(
        question=question,
        schema_summary=schema_summary,
        failure_reason=failure_reason,
    )
    return [{"role": "user", "content": user_content}], CLARIFICATION_SYSTEM
