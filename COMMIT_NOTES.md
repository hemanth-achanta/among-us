## Multi-table RCA & Senior Analyst Upgrade

- **Root cause fix**: Increased `SCHEMA_CONTEXT_TOKEN_LIMIT` from 3,000 to 8,000 tokens — the LLM was literally unable to see all 3 tables because the schema was truncated at 12K chars. Now all tables, relationships, and join notes fit within the 32K char budget.
- **Added table relationships and join notes** to `schema.yaml`: explicit mappings between orders ↔ metrics (via `order_id`), orders ↔ sessions (via comma-separated `order_ids` with Presto UNNEST syntax), plus a table selection guide for when to use each table.
- **Overhauled all LLM prompts** for analyst-grade output:
  - SQL generation: Presto/Trino syntax guide, RCA reasoning patterns, multi-table JOIN guidance, partition filter reminders.
  - Result interpretation: Removed the "3-5 sentences" cap; now produces thorough RCA-style analysis with specific numbers, pattern identification, and proactive insights.
  - Query planning: Smarter about when to JOIN tables in a single query vs. separate queries; explicit RCA strategy (quantify → break down → drill in).
  - Complexity estimation: Added RCA/funnel/conversion/session signals to heuristic detection.
- **Fixed SQL validator** for multi-table queries: multi-CTE regex now handles `WITH cte1 AS (...), cte2 AS (...)`, and Presto's `UNNEST`/`LATERAL` are no longer flagged as unknown tables.
- **Schema loader** now injects relationships and join notes into every SQL generation prompt, and filters to only relevant relationships when using schema overrides.
- **Result interpreter** now computes summary statistics (sum/mean/min/max) for numeric columns and passes them to the LLM alongside raw data for richer analysis.
- **Config tuning**: `MAX_QUERY_ITERATIONS` 3→5, `MAX_QUERIES_PER_QUESTION` 3→4, `SCHEMA_SAMPLE_ROWS` 3→2 (token cost savings), planner `max_tokens` 256→512.

Run status: N
