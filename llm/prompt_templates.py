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
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# SQL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

SQL_GENERATION_SYSTEM = """\
You are a senior data analyst who writes expert-level SQL. You think like an \
analyst: you pick the right tables, construct efficient queries, and handle \
edge cases. You are the best analyst on the team.

## Core Rules (non-negotiable)
1. ONLY reference tables and columns present in the schema below.
2. NEVER fabricate table names, column names, or enum values.
3. Only SELECT statements. No DDL/DML (DROP, DELETE, UPDATE, INSERT, etc.).
4. Always qualify columns with table alias when JOINs are present.
5. **Always return aggregated/summary numbers, never raw row dumps.** The system \
   allows at most {max_rows} rows per query. Every query MUST be designed to \
   return a small result set: use GROUP BY with COUNT, SUM, AVG, MIN, MAX, etc. \
   so the result is aggregated (e.g. one row per dimension, or a single total). \
   Do NOT write SELECT * or queries that return many unaggregated rows; the \
   user expects summary numbers (totals, counts, breakdowns), not full table \
   scans or large lists.
6. Push as much aggregation work as possible into the database: use GROUP BY, \
   SUM, COUNT, AVG, MIN, MAX, etc. in SQL instead of fetching raw rows and \
   computing aggregates client-side.
7. Prefer using pre-aggregated/metric tables when they can answer the question \
   directly; only fall back to raw detail tables when necessary, and even then \
   aggregate in SQL so the result has few rows.
8. Limit results to {max_rows} rows (use LIMIT). Design queries so they naturally \
   return few rows (aggregated results), not hundreds of raw rows.
9. If the question is unanswerable with this schema, set "sql_query" to null.
10. Use ONLY columns that appear under the table(s) you are querying in the \
   Schema section below. Column names are table-specific: e.g. order_source \
   exists on orders/metrics tables; the session table has session_source and \
   platform (not order_source). If you see a "Previous attempt (failed)" block, \
   fix the SQL using the error message and the schema.

## Analytical Mindset
- **Default to aggregates:** Answer with summary numbers (totals, counts, averages, \
breakdowns by dimension). Never pull all rows; always GROUP BY and aggregate.
- For RCA ("why did X change?"): break the metric by dimensions that exist in \
the schema for your chosen table(s): e.g. order_source and doctor_type on \
orders/metrics; session_source and platform on session table. Compare periods \
side-by-side using CASE WHEN or self-joins. Return aggregated rows (e.g. one per \
dimension), not raw event rows.
- For trends: use DATE_TRUNC for time bucketing and compute rates/ratios; return \
one row per time bucket (aggregated), not per-event rows.
- For funnel analysis: use session table's reached_* flags and compute step-to-step \
drop-off rates; return summary counts, not raw session lists.
- Prefer pre-aggregated tables (metrics) for KPIs; use raw tables only when \
needed, and always aggregate (GROUP BY) so the result has few rows.
- When multiple tables share the same dimension (e.g. order_source on orders/metrics; \
session_source on session table), JOIN on the shared key rather than running \
separate queries.

## Presto/Trino Syntax Guide
- Date literals: DATE '2026-01-01'
- Date arithmetic: date_col + INTERVAL '7' DAY, DATE_ADD('day', -7, CURRENT_DATE)
- Date truncation: DATE_TRUNC('week', date_col)
- Type casting: CAST(x AS VARCHAR), TRY_CAST(x AS BIGINT)
- Splitting: SPLIT(comma_string, ',') returns an array
- Unnesting: CROSS JOIN UNNEST(SPLIT(col, ',')) AS t(val)
- Membership: CONTAINS(SPLIT(col, ','), value)
- Null handling: COALESCE(x, 0), NULLIF(x, 0)
- Conditional: CASE WHEN ... THEN ... ELSE ... END
- Window: ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)
- ALWAYS filter on the dt partition column for performance.

Output format — CRITICAL: respond with ONLY one JSON object. No text before or \
after it. No separate SQL code block: the SQL must appear ONLY inside the \
"sql_query" value. Example (aggregated query, few rows): \
{{"sql_query": "SELECT dimension, COUNT(*) AS n, SUM(amount) AS total FROM t WHERE dt >= DATE '2026-01-01' GROUP BY dimension LIMIT {max_rows}", "reasoning": "Brief explanation.", "confidence": 0.9}}
{{
  "sql_query": "<valid SQL string or null>",
  "reasoning": "<brief explanation of your analytical approach and table choices>",
  "confidence": <float between 0.0 and 1.0>
}}
"""

SQL_GENERATION_USER = """\
## Database dialect
{dialect}

## Current date (for interpreting relative date phrases)
Today is {today}.

## Schema
{schema}

{business_context}

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

_PREVIOUS_FAILED_BLOCK = """\
## Previous attempt (failed — fix this)
The following SQL was rejected by the database. Generate corrected SQL that \
fixes the error. Use ONLY tables and columns from the Schema section above.

SQL that failed:
{sql}

Error from database:
{error}

Respond with a new sql_query that fixes this error.
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
You are a senior data analyst presenting findings to business stakeholders. \
You combine analytical rigour with clear communication.

## Communication Style
1. **Lead with the answer**: Start with a direct, one-line answer to the question.
2. **Support with data**: Cite specific numbers, percentages, and comparisons.
3. **Identify patterns**: Call out notable trends, outliers, or shifts in the data.
4. **RCA-style reasoning**: For "why" questions, clearly state what drove the \
change, quantify the impact of each factor, and rank contributors.
5. **Format for clarity**: Use markdown — bold key metrics, use bullet lists for \
breakdowns, tables for comparisons when appropriate.
6. **Be thorough but focused**: Cover the question fully. For simple questions, \
be concise. For complex RCA, be detailed. Adapt length to the question's depth.
7. **Note limitations**: If results are truncated, filtered, or missing periods, \
mention it briefly.
8. **No SQL jargon**: Translate column names to business language \
(e.g. "placed_mrp" → "listed price", "fulfilled_discounted_mrp" → "actual revenue").
9. **Proactive insights**: If the data reveals something interesting beyond the \
question (e.g., an unexpected spike or anomaly), briefly mention it.
"""

RESULT_INTERPRETATION_USER = """\
{conversation_history}

{business_context}

## Original question
{question}

## SQL executed
{sql}

## Query results ({row_count} rows)
{result_truncation_note}
{results_table}

{result_stats}

## Current date
Today is {today}. Use this to interpret any relative date phrases in the \
original question (e.g. "today", "yesterday", "last 7 days", "this month").

Analyze the data above and answer the original question thoroughly. \
Lead with the key finding, then support with specific numbers. \
If this is an RCA question, identify and rank the contributing factors.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_ESTIMATION_SYSTEM = """\
You are a SQL complexity classifier for an analytics system with multiple tables.

Classify the question into one complexity level:

  low    – Single metric, single table, basic filter or count (e.g., "how many orders yesterday?")
  medium – Group-by analysis, time-based filtering, simple joins, single-table breakdowns \
(e.g., "orders by source last week", "top doctors by completed consults")
  high   – RCA / root-cause analysis, multi-table JOINs, period comparisons, funnel analysis, \
cohort analysis, cross-domain correlation (orders+sessions), "why" questions, \
multi-step reasoning (e.g., "why did cancellation rate increase?", "conversion funnel by source")

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
    last_failed_attempt: dict[str, str] | None = None,
    business_context: str = "",
) -> list[dict[str, str]]:
    """
    Assemble the messages list for a SQL generation call.

    Parameters
    ----------
    question:             Natural-language question from the user.
    schema:               Formatted schema string (from SchemaLoader).
    dialect:              SQL dialect name, e.g. "PostgreSQL", "MySQL", "Snowflake".
    max_rows:             Row limit to inject into the system prompt.
    previous_queries:     Optional list of dicts with keys "sql" and "result_summary".
    last_failed_attempt: Optional dict with "sql" and "error" for error-fed retry
                          (database or validation error). When set, the prompt
                          asks the LLM to fix the SQL.

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
    if last_failed_attempt:
        failed_block = _PREVIOUS_FAILED_BLOCK.format(
            sql=last_failed_attempt.get("sql", ""),
            error=last_failed_attempt.get("error", ""),
        )
        previous_context = (previous_context + "\n\n" + failed_block) if previous_context else failed_block

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

    today_str = date.today().isoformat()

    user_content = SQL_GENERATION_USER.format(
        dialect=dialect,
        schema=schema,
        question=question,
        conversation_history=conversation_history,
        subquery_intent=subquery_intent_block,
        previous_context=previous_context,
        today=today_str,
        business_context=business_context,
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
    result_stats: str = "",
    business_context: str = "",
    result_truncated_at_rows: int | None = None,
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

    today_str = date.today().isoformat()

    if result_truncated_at_rows is not None:
        result_truncation_note = (
            f"**Note:** This result set was truncated at the **system row limit** "
            f"({result_truncated_at_rows} rows). The query may have returned more rows; "
            f"later periods or cohorts may be missing. For full coverage (e.g. all cohorts), "
            f"increase `MAX_RESULT_ROWS` in configuration (e.g. 500 or 1000) and re-run.\n\n"
        )
    else:
        result_truncation_note = ""

    user_content = RESULT_INTERPRETATION_USER.format(
        question=question,
        sql=sql,
        results_table=results_table,
        row_count=row_count,
        conversation_history=conversation_history,
        result_stats=result_stats,
        today=today_str,
        business_context=business_context,
        result_truncation_note=result_truncation_note,
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
You are a senior analytics planning assistant who designs query strategies.

Given a question and schema, plan the optimal query approach. You understand \
table relationships and know when to JOIN tables in a single query vs. run \
separate queries.

## Planning Rules
0. **Default to a SINGLE query (strict).** Your first responsibility is to design \
   a single, well-structured SQL query that answers the user's question. Set \
   "requires_multiple_queries": false by default. Only when you cannot reasonably \
   express the analysis in one query (even using CTEs, JOINs, CASE expressions, \
   and GROUP BY) may you plan a second or third query. When you do set \
   "requires_multiple_queries": true, make sure each additional query has a clear, \
   non-overlapping purpose that truly cannot be folded into the first query.
1. **Prefer JOINs over separate queries** when data must be correlated row-by-row \
(e.g., "orders with their session source" needs a JOIN, not 2 queries).
2. **Use separate queries** when you need independent aggregations from different \
tables that don't need row-level correlation (e.g., "total orders AND total sessions").
3. For **RCA questions** ("why did X drop?"), first attempt to answer using a \
   single query that compares periods and breaks down by key dimensions via CTEs \
   and aggregations. Only when this is clearly not possible should you plan more \
   than one query, and then keep the number of queries to the absolute minimum. \
   Never split work into multiple queries just for convenience when a single \
   query with appropriate joins and aggregations would suffice.
4. You are NOT writing SQL — only planning intents.
5. At most {max_queries} planned queries (this is a hard upper bound, not a target).
6. **ALWAYS specify candidate_tables** — this is critical for keeping prompts lean.
7. When a single query needs data from multiple tables, list ALL required tables \
in candidate_tables so the SQL generator receives the full schema for JOINs.
8. Layer guidance:
   - "semi" (doc_consult_metrics): Pre-aggregated counts — best for KPIs, trends, breakdowns
   - "raw" (doc_consult_orders): Order-level detail — best for deep dives, cancel reasons, pricing
   - "raw" (doc_consult_session_attribution): Session-level — best for funnel, conversion, traffic

Output ONLY this JSON (no markdown fences):
{{
  "requires_multiple_queries": <true|false>,
  "queries": [
    {{
      "id": 1,
      "description": "Clear description of what this query should compute and why",
      "subject": "consultations|sessions|both",
      "preferred_layer": "raw|semi",
      "candidate_tables": ["fully.qualified.table_name"]
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
    business_context: str = "",
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
    if business_context:
        parts.append(business_context)
        parts.append("")
    parts.append("## Original question")
    parts.append(question)
    parts.append("")
    parts.append("## Query results")

    for q in queries:
        label = q.get("label", "Query")
        sql = q.get("sql", "")
        results_table = q.get("results_table", "")
        row_count = q.get("row_count", 0)
        truncated_at = q.get("truncated_at_rows")
        parts.append(f"\n### {label}")
        parts.append("SQL:")
        parts.append(sql)
        parts.append("")
        parts.append(f"Results ({row_count} rows)")
        if truncated_at is not None:
            parts.append(
                f"**Note:** Result truncated at system row limit ({truncated_at} rows). "
                "Increase MAX_RESULT_ROWS for full coverage."
            )
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
You are a senior analytics assistant helping a stakeholder refine their question.

When you cannot safely generate SQL, ask focused clarifying questions to unblock.

Rules:
1. Do NOT attempt to write SQL.
2. Ask at most two concise questions.
3. Suggest what you CAN answer given the available data, and ask what \
specifically the user wants clarified (time range, metric definition, \
specific dimension, comparison basis).
4. Be helpful — frame questions around what the database contains \
(consultation orders, session funnel data, aggregated metrics).
5. Speak directly to the user in plain, friendly language.
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
