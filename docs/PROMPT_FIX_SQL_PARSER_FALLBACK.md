# Prompt: Fix SQL generation when LLM returns prose + code block

Copy the text below and give it to the Cursor agent to implement the fix.

---

Fix the SQL generation flow so that when the LLM returns prose and a ` ```sql ... ``` ` block instead of the required JSON, we still run the query instead of dropping it.

**Problem:** For some subqueries (e.g. session funnel using `doc_consult_session_attribution`), the LLM responds with explanatory text and SQL in a ` ```sql ... ``` ` block instead of a single JSON object `{"sql_query": "...", "reasoning": "...", "confidence": ...}`. The parser in `backend/sql_generator.py` only handles JSON; on parse failure it returns `sql_query=None`, so that query is never executed and its results (e.g. session metrics) never appear in the final answer.

**Required changes:**

1. **`llm/prompt_templates.py`**  
   In the SQL generation system prompt (around `SQL_GENERATION_SYSTEM` / "Output format"):  
   - State explicitly: respond with **only** one JSON object; no text before or after, and no separate SQL code block—the SQL must appear only inside the `sql_query` value.  
   - Optionally add a one-line example of the exact JSON shape.

2. **`backend/sql_generator.py`**  
   In `_parse_response` (or equivalent): when JSON parsing fails (e.g. `json.loads` raises or we don't get a valid `sql_query`):  
   - Try a **fallback**: detect a markdown SQL code block (e.g. ` ```sql ... ``` ` or ` ```\n...SELECT... ``` `) in the raw response.  
   - If found, set `sql_query` to that extracted SQL, set `reasoning` to a short note like "Extracted from code block (LLM did not return valid JSON)", and set `confidence` to a default (e.g. 0.7).  
   - Return that so the orchestrator still executes the query.  
   - Log a warning when this fallback is used (e.g. via the existing logger) so we can monitor it.

After the fix, a response that is prose + ` ```sql ... ``` ` should still result in the SQL being executed and its results included in the final answer (e.g. session metrics when the session table was used).

---
