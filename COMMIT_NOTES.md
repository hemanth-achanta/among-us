## Changes in this commit

- **API cost in debug mode** — Estimated cost (USD) for each question and for the session is now shown when "Show debug details" is on. Config adds `MODEL_PRICING` (per-model USD per 1M input/output tokens). `TokenUsageSummary` tracks per-call usage and exposes `cost_usd(pricing)`; orchestrator includes `cost_usd` in `token_summary`. Sidebar shows "API cost" for the session; each answer’s debug panel shows "API cost" for that question.

- **Live status feed** — While the pipeline runs, the last 3 pipeline steps are shown in real time in small font, one below the other (current step bold with ⏳, completed steps with ✅). Implemented via a new `status_callback` in `orchestrator.run()` and `_run_single_iteration()`, fired at each stage (complexity, model selection, query planning, SQL generation, validation, execution, interpretation). The feed is always visible (not only in debug mode) and clears when the answer is ready.

- **SQL generation: aggregates only, no raw row dumps** — Prompt in `llm/prompt_templates.py` updated so the LLM must return aggregated/summary numbers only, never large raw row dumps. New rule states the system allows at most 500 rows per query and requires GROUP BY with COUNT/SUM/AVG etc.; forbids SELECT * or queries that return many unaggregated rows. Analytical mindset now defaults to aggregates, and the example in the prompt shows an aggregated query pattern with LIMIT.

- **README** — Added project README with setup, env/config, run instructions, debug mode and API cost, and project layout.
