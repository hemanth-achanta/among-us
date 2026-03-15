# Analytics Assistant

LLM-powered text-to-SQL analytics: ask questions in plain English and get answers backed by your database. The assistant generates SQL, runs it (with validation), and returns a natural-language summary of the results.

## Features

- **Multi-database support**: Trino/Presto (default), PostgreSQL, MySQL, Snowflake
- **Claude API**: Complexity-based model routing (Haiku for planning, Sonnet for SQL and interpretation) with retries and escalation
- **Schema-driven**: YAML schema with table/column descriptions and optional sample rows; token-budget–aware context
- **Streamlit UI**: Chat interface, optional debug panel (SQL, token usage, **API cost per question**), conversation traces
- **Conversation logging**: Optional per-question JSON traces under `logs/conversations/` for debugging and auditing

## Prerequisites

- Python 3.10+
- Access to a supported database (Trino/Presto, PostgreSQL, MySQL, or Snowflake)
- [Anthropic API key](https://console.anthropic.com/) for Claude

## Setup

1. **Clone and install dependencies**

   ```bash
   cd among-us
   pip install -r requirements.txt
   ```

2. **Environment**

   Copy or create a `.env` in the project root. Required and common variables:

   ```bash
   # Required
   ANTHROPIC_API_KEY=sk-ant-...
   DB_USER=your_user
   DB_PASSWORD=your_password

   # Database (default: Trino)
   DATABASE_TYPE=trino
   DB_HOST=your-trino-host
   DB_PORT=8443

   # Optional
   DB_NAME=lakehouse
   DB_SCHEMA=data_model
   SCHEMA_FILE_PATH=/path/to/schema.yaml
   SHOW_DEBUG_INFO=true
   ```

   See `config/config.py` for all options (logging, models, retries, DeepAnalyze, etc.).

3. **Schema**

   Place a schema YAML at the path given by `SCHEMA_FILE_PATH` (default: `schema.yaml` in the project root). It should define `tables` and optionally `relationships`; see `schema_example.yaml` for structure.

## Run

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`). Use the sidebar to confirm “Pipeline ready”, then ask a question in the chat.

## Debug mode and API cost

- **Show debug details**: In the sidebar, enable “Show debug details” to see per-answer SQL, query plan, model, iterations, token usage, and **estimated API cost (USD)** for that question.
- **Session cost**: The sidebar also shows cumulative input/output tokens and **total API cost** for the session. Cost is computed from token usage and per-model pricing in `config.config.MODEL_PRICING` (Anthropic list prices; adjust there if needed).

## Project layout

- `app/streamlit_app.py` — Streamlit UI and pipeline wiring
- `backend/` — Query orchestration, SQL generation/validation/execution, result interpretation, complexity estimation
- `config/config.py` — Central config (env + defaults)
- `db/` — Connection managers and schema loader
- `llm/llm_client.py` — Anthropic client, token usage, and cost estimation
- `utils/` — Logging, conversation trace logger
- `schema.yaml` / `schema_example.yaml` — Table definitions for the LLM

## License

Use and modify as needed for your organisation.
