# Cursor prompt: Generate schema.yaml from repo analysis

Use this prompt in Cursor after adding the GitHub repo (or local folders) that contain SQL, notebooks, reports, and pipelines. **The repo is context only** — use it to understand how columns are used, but **generate schema only for the tables you explicitly ask for** in your message.

---

## Prompt (copy below)

```
I need you to generate schema.yaml entries (in this project's schema.yaml format) **only for the tables I ask for below**. Use the repo/folders I've shared purely as context to understand how those tables and their columns are used — do not generate schema for every table in the repo.

**Tables I want schema for:** [LIST THE TABLE(S) HERE, e.g. hive.data_model.doc_consult_orders and hive.pre_analytics.doc_consult_metrics]

**What to do:**
1. **Use the shared repo/folders as context only.** Scan SQL files, notebooks, report definitions, and pipeline code to see how the **requested table(s)** and their columns are used (filters, joins, aggregations, reports). Do not output schema for any other tables.

2. **For each requested table**, produce:
   - **description** — table-level, reflecting how it's used in the repo.
   - **columns** — each with **type** (VARCHAR, INTEGER, BIGINT, DOUBLE, DATE, TIMESTAMP, etc. where visible) and **description** that reflects **actual usage** in the repo, e.g.:
     - "Used in funnel reports to filter by channel (e.g. Mobile_Website, Android_App)"
     - "Joined with doc_consult_orders in pre_analytics; used for cohort breakdowns"
     - "Aggregated in dashboards as sum(placed_mrp)"
   Prefer usage-based context over generic definitions.

3. **Relationships and join_notes** — only between the tables you were asked for (and only if you were asked for more than one). Infer from JOINs and shared keys in the repo. Include any special join logic (e.g. comma-separated IDs, partition filters).

4. **Output format** must match this project's schema.yaml exactly:
   - Under `tables`, include only the requested table(s), each with: `description`, optional `layer`, optional `subject`, `columns` (each with `type` and `description`), optional `sample_rows` if you find examples in the repo.
   - Column descriptions can wrap with 2-space indent.
   - If multiple tables requested: add `relationships` (from_table, from_column, to_table, to_column, type, description) and `join_notes` (multi-line string) only for those tables.
   - Use the existing schema.yaml in this workspace as the exact structural reference.

Return the schema content for the requested tables only. Say if the repo didn't reference a requested table or column so I can add descriptions manually.
```

---

## How to use

1. **In Cursor:** Open the project that contains your `schema.yaml` (this among-us project).
2. **Add context:** Use @ to include the GitHub repo or local folders (SQL, notebooks, reports, pipelines).
3. **In the prompt above,** replace `[LIST THE TABLE(S) HERE, ...]` with the exact table(s) you want, e.g.:
   - `hive.data_model.doc_consult_orders`
   - `hive.data_model.doc_consult_orders and hive.pre_analytics.doc_consult_metrics`
4. **Send the prompt.** Cursor will use the repo only to enrich understanding and will return schema **only for those tables**.
5. **Merge or paste** the output into your schema.yaml (or a new file) and adjust as needed.
