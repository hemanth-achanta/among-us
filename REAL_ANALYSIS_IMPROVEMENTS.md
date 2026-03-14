# Evaluation: Making the App Ready for Real Analysis

This document summarizes gaps and improvements so the analytics assistant can reliably handle **real** analysis questions (complex filters, joins, time ranges, business metrics, and multi-step reasoning).

---

## 1. What the LLM Currently Knows

| Context | Source | Notes |
|--------|--------|--------|
| **Tables & columns** | `schema.yaml` → `format_for_prompt()` | Table names, column types, descriptions. |
| **Sample rows** | `schema.yaml` (per table) | Up to `SCHEMA_SAMPLE_ROWS` (default 3) per table; helps with value examples. |
| **Layers & subjects** | `schema.yaml` (`layer`, `subject`) | Used in planning (raw / semi / processed) and layered summary. |
| **Schema summary** | `format_summary_for_complexity()` | Table names + column names only (no descriptions). |
| **Layered summary** | `format_layered_summary()` | Tables grouped by layer and subject for query planning. |
| **Dialect** | `connection_manager.dialect_name` | Currently `"Presto SQL (Trino)"` — minimal. |
| **Previous query results** | `summarise_result()` | First 10 rows, ~1000 chars; passed as “previous context” in follow-up generations. |
| **Chat history** | Last N turns (`CHAT_CONTEXT_TURNS`) | Passed to SQL generation and interpretation. |

---

## 2. Gaps That Block Real Analysis

### 2.1 Schema & Data Context

- **No explicit relationships**
  - `schema.yaml` has `relationships: []`. The LLM must infer JOINs from column names (e.g. `customer_id`) and descriptions. For real analysis across `doc_consult_orders`, `doc_consult_metrics`, and `doc_consult_session_attribution`, explicit FK-style hints (e.g. `doc_consult_orders.order_id → doc_consult_metrics.order_id`) would reduce wrong joins and hallucinated links.

- **No business glossary / key metrics**
  - Terms like “consultation”, “order”, “session”, “conversion”, “MRP”, “PLD” are domain-specific. A short glossary (e.g. “MRP = Maximum Retail Price; use `placed_mrp` or `fulfilled_mrp`”) and a “key metrics” section (which columns are KPIs, which tables are best for “count of consultations” vs “revenue”) would improve consistency.

- **Dialect is under-specified**
  - The prompt only says “Presto SQL (Trino)”. For Trino you want explicit hints: use `LIMIT` (not `TOP`), prefer `date_trunc('month', col)` / `DATE(col)` where relevant, and that fully qualified names are `catalog.schema.table` (e.g. `hive.data_model.doc_consult_orders`). Adding a small “Trino rules” block in the SQL prompt would cut dialect mistakes.

- **Partition / freshness not documented**
  - Many analytics tables are partitioned (e.g. `dt`). For “latest data” or “last 30 days”, the LLM needs to know:
    - Which column is the partition/date (e.g. `dt`, `order_placed_date`).
    - That filtering on partition (e.g. `dt >= current_date - 30`) is important for performance and correctness.
  - Optional: add a short “data freshness” or “partition key” hint per table in the schema or in the prompt.

- **Token budget can truncate schema**
  - `SCHEMA_CONTEXT_TOKEN_LIMIT` (default 3000) can truncate the schema string. For large schemas, the LLM might not see all tables. Mitigations: increase budget where possible, or add **schema trimming by relevance** (e.g. use planning’s `candidate_tables` more aggressively, or a lightweight retrieval step that selects tables by question).

### 2.2 Pipeline & Retry Behavior

- **Retry on low confidence drops context**
  - In `query_orchestrator._run_single_iteration`, when `retry_mgr.needs_retry_for_confidence(gen_result.confidence)` is true, the retry call to `generate()` does **not** pass `chat_history`, `subquery_description`, or `schema_override`. So the retry uses a weaker context and may repeat the same mistake. **Fix:** pass the same arguments as the first call (including `chat_history`, `subquery_description`, `schema_override`).

- **Previous-query summary may be too small for complex analysis**
  - `summarise_result()` uses ~1000 chars and 10 rows. For multi-step analysis (e.g. “compare two segments then summarize”), the next query might need more context (e.g. key aggregates or a short summary of the first result). Consider a slightly larger budget for “previous context” when `len(previous_queries) > 0`, or allow a brief “summary of what we learned” in addition to the raw snippet.

### 2.3 Interpretation & Results

- **Interpretation table cap**
  - Result interpreter uses `_MAX_TABLE_CHARS = 2_000` and `df.head(50)`. For “real” analysis with many rows or columns, the model might not see the full picture. Options: increase cap for interpretation only, or add a “summary stats” line (min/max/mean for numeric columns) when the result is large.

- **No guidance for “no data”**
  - When the query returns 0 rows, the interpreter uses a generic message. For real analysis, it can help to suggest: “Try broadening the time range or relaxing filters (e.g. status, region).” This could be a one-line addition to the “no results” prompt or a small rule in the interpreter.

### 2.4 Planning & Multi-Query

- **Planner has no explicit “key metrics” or “recommended tables”**
  - The planner sees table names, layers, and short descriptions. Adding a few lines like “For ‘consultation count’ prefer `doc_consult_metrics` (semi); for order-level detail use `doc_consult_orders` (raw)” would steer multi-query and single-query choices.

- **Candidate tables not validated**
  - The planner can return `candidate_tables` that might not exactly match schema keys (e.g. casing, qualification). The code already normalizes to known tables when building `schema_override`, but ensuring planner output is validated against `schema_loader.table_names` would avoid rare edge cases.

---

## 3. What to Add So the LLM “Knows More”

1. **Relationships**
   - Populate `relationships` in `schema.yaml` (manually or via `schema_definer` / DB metadata). Format already supported: e.g. `doc_consult_orders.order_id → doc_consult_metrics.order_id`. Emit this block in `format_for_prompt()` (already done when non-empty).

2. **Business glossary / key metrics (optional section in schema or prompt)**
   - Short list of terms and recommended columns/tables, e.g.:
     - “Consultation order = one row in doc_consult_orders; for counts use completed_consult_orders in doc_consult_metrics when possible.”
     - “MRP = Maximum Retail Price; placed_mrp / fulfilled_mrp.”
   - Could be a new top-level key in `schema.yaml` (e.g. `glossary`, `key_metrics`) and included in the SQL generation user prompt.

3. **Trino-specific rules in the SQL prompt**
   - Add a small “Dialect rules” block: e.g. use `LIMIT n`, qualify tables as `catalog.schema.table` when needed, and use Trino date functions. This reduces TOP/FETCH FIRST and wrong qualifier mistakes.

4. **Partition / date columns**
   - In table descriptions or a shared “Data freshness” note: “For time-bound questions, filter on `dt` (partition) or `order_placed_date` where relevant. Prefer `dt >= current_date - interval '30' day` for recent data.”

5. **Retry context**
   - Pass `chat_history`, `subquery_description`, and `schema_override` into every `generate()` call, including the low-confidence retry path.

6. **Clarification prompt**
   - When asking clarifying questions, use the **layered** schema summary (or a short summary that includes layer/subject and key tables) so the LLM can suggest “e.g. do you want order-level detail (raw) or pre-aggregated metrics (semi)?”.

---

## 4. Summary: Priority Order

| Priority | Change | Impact |
|----------|--------|--------|
| **High** | Add **relationships** to `schema.yaml` (or generate them) | Fewer wrong JOINs; better multi-table analysis. |
| **High** | **Retry:** pass `chat_history`, `subquery_description`, `schema_override` in low-confidence retry | More consistent and context-aware retries. |
| **Medium** | **Dialect:** add Trino-specific rules (LIMIT, qualification, dates) to SQL prompt | Fewer dialect errors. |
| **Medium** | **Partition/freshness:** document `dt` (and other date keys) and suggest filtering for “latest” / “last N days” | Correct and efficient time-bound analysis. |
| **Medium** | **Glossary / key metrics:** short business terms + recommended tables/columns | More consistent metric and table choice. |
| **Low** | Slightly larger previous-query summary when multi-query | Better multi-step reasoning. |
| **Low** | Optional schema trimming by relevance for very large schemas | Avoid truncation and focus context. |
| **Low** | Richer “no results” interpretation (suggest broadening filters/time) | Better UX for empty result sets. |

---

## 5. Conclusion

The app is already well structured (planning, layers, validation, clarification). To handle **real** analysis questions reliably:

- **Give the LLM more structure:** relationships, dialect rules, partition/date hints, and optionally a small business glossary/key metrics section.
- **Fix the retry context** so the model always has full context (chat, subquery intent, schema override) on retries.
- **Tighten interpretation and empty-result handling** so answers stay grounded and actionable.

Implementing the high- and medium-priority items above will significantly improve correctness and consistency for real-world analysis requests.
