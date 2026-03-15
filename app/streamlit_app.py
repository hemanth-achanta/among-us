"""
Streamlit frontend for the Analytics Assistant.

Layout
------
• Sidebar  — configuration, schema status, token usage, debug toggle.
• Main     — chat-style message history; question input at the bottom.

Each answer displays:
  1. The natural-language answer.
  2. (Optional debug panel) Generated SQL, query plan, model used, iterations,
     token usage, and per-step details.

The heavy pipeline objects (LLMClient, ConnectionManager, SchemaLoader,
QueryOrchestrator) are created once and cached in ``st.session_state`` so they
survive reruns.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Make the project root importable regardless of launch directory ───────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config
from config.config import validate_config
from utils.logger import configure_logging, get_logger, Timer
from app.business_context import (
    add_suggestion,
    get_approved_context,
    format_approved_for_prompt,
)

# Optional: DeepAnalyze local report generation
try:
    from backend.deepanalyze_client import generate_report, DeepAnalyzeError
except ImportError:
    generate_report = None  # type: ignore[misc, assignment]
    DeepAnalyzeError = Exception  # type: ignore[misc, assignment]

# Bootstrap logging before anything else
configure_logging(
    level=config.LOG_LEVEL,
    fmt=config.LOG_FORMAT,
    log_file=config.LOG_FILE,
)

log = get_logger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session-state keys with sensible defaults."""
    defaults = {
        "messages":        [],      # list of {"role": "user"|"assistant", "content": ..., "meta": ...}
        "orchestrator":    None,
        "schema_loader":   None,
        "llm_client":      None,
        "conn_manager":    None,
        "init_error":      None,
        "show_debug":      config.SHOW_DEBUG_INFO,
        # DeepAnalyze report: last query result and generated report
        "last_result_df":       None,
        "last_result_question": "",
        "last_result_sql":      None,
        "last_deepanalyze_report": None,
        "last_deepanalyze_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _build_pipeline(schema_def: object) -> None:
    """
    Instantiate all heavy pipeline components and store them in session state.

    Uses ``make_connection_manager()`` to automatically select the right backend:
      - ``DATABASE_TYPE=trino``      → PrestoConnectionManager (notebook pattern)
      - ``DATABASE_TYPE=postgresql`` → ConnectionManager (SQLAlchemy)
      - ``DATABASE_TYPE=mysql``      → ConnectionManager (SQLAlchemy)
      - ``DATABASE_TYPE=snowflake``  → ConnectionManager (SQLAlchemy)

    Called once per session (or when the user clicks "Re-initialise").
    Errors are captured and surfaced in the UI.
    """
    from db.connection_manager import make_connection_manager
    from db.schema_loader import SchemaLoader
    from llm.llm_client import LLMClient
    from backend.query_orchestrator import QueryOrchestrator

    try:
        validate_config()

        conn_manager = make_connection_manager()

        with st.spinner(
            f"Connecting to {conn_manager.dialect_name}…"
        ):
            if not conn_manager.health_check():
                st.session_state["init_error"] = (
                    f"Database health check failed ({conn_manager.dialect_name}). "
                    "Verify DB_HOST, DB_PORT, DB_USER, DB_PASSWORD in .env."
                )
                return

        schema_loader = SchemaLoader(schema_def)
        llm_client    = LLMClient()
        orchestrator  = QueryOrchestrator(
            schema_loader=schema_loader,
            connection_manager=conn_manager,
            llm_client=llm_client,
        )

        st.session_state["conn_manager"]  = conn_manager
        st.session_state["schema_loader"] = schema_loader
        st.session_state["llm_client"]    = llm_client
        st.session_state["orchestrator"]  = orchestrator
        st.session_state["init_error"]    = None

        log.info(
            "pipeline_built_successfully",
            db_backend=conn_manager.dialect_name,
        )

    except Exception as exc:  # noqa: BLE001
        st.session_state["init_error"] = str(exc)
        log.error("pipeline_build_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Schema definition
# ─────────────────────────────────────────────────────────────────────────────

def _default_schema() -> dict:
    """
    Default schema for the Trino ``lakehouse`` catalog.

    Tables are fully qualified as ``lakehouse.<schema>.<table>`` so the LLM
    generates correctly qualified SQL for Trino.

    Replace sample_rows with real rows once the catalog is explored, or load
    the full schema from a YAML file:  SchemaLoader("schema.yaml").
    """
    return {
        "tables": {
            "lakehouse.datamodel.doc_consult_orders": {
                "description": (
                    "Consulting/doctor order records from the data model layer. "
                    "Replace column definitions with actual columns from the table."
                ),
                "columns": {
                    "order_id":    {"type": "VARCHAR",   "description": "Unique order identifier"},
                    "doctor_id":   {"type": "VARCHAR",   "description": "Attending doctor identifier"},
                    "patient_id":  {"type": "VARCHAR",   "description": "Patient identifier"},
                    "order_date":  {"type": "DATE",      "description": "Date the order was placed"},
                    "order_type":  {"type": "VARCHAR",   "description": "Type/category of the order"},
                    "status":      {"type": "VARCHAR",   "description": "Current order status"},
                    "quantity":    {"type": "BIGINT",    "description": "Quantity ordered"},
                    "amount":      {"type": "DOUBLE",    "description": "Order amount"},
                    "created_at":  {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                },
                "sample_rows": [],  # populated once connectivity is confirmed
            }
        },
        "relationships": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        st.caption("LLM-powered Text-to-SQL analytics")

        st.divider()

        # ── Status ───────────────────────────────────────────────────────────
        st.subheader("Status")
        if st.session_state["init_error"]:
            st.error(f"Init error: {st.session_state['init_error']}")
        elif st.session_state["orchestrator"] is not None:
            st.success("Pipeline ready")
            cm = st.session_state.get("conn_manager")
            if cm:
                st.caption(f"🗄️ Backend: **{cm.dialect_name}**")
                st.caption(f"🌐 Host: `{config.DB_HOST}`")
            if st.session_state["schema_loader"]:
                tables = st.session_state["schema_loader"].table_names
                with st.expander("📋 Schema tables", expanded=False):
                    for t in tables:
                        st.code(t, language=None)
        else:
            st.warning("Pipeline not initialised")

        st.divider()

        # ── Debug toggle ─────────────────────────────────────────────────────
        st.subheader("Options")
        st.session_state["show_debug"] = st.toggle(
            "Show debug details",
            value=st.session_state["show_debug"],
            help="Display generated SQL, model, iterations, and token usage.",
        )

        st.divider()

        # ── Token usage and cost ─────────────────────────────────────────────
        if st.session_state["messages"]:
            st.subheader("Session token usage")
            total_in  = sum(
                m.get("meta", {}).get("token_summary", {}).get("input_tokens",  0)
                for m in st.session_state["messages"] if m["role"] == "assistant"
            )
            total_out = sum(
                m.get("meta", {}).get("token_summary", {}).get("output_tokens", 0)
                for m in st.session_state["messages"] if m["role"] == "assistant"
            )
            total_cost_usd = sum(
                m.get("meta", {}).get("token_summary", {}).get("cost_usd", 0) or 0
                for m in st.session_state["messages"] if m["role"] == "assistant"
            )
            currency = getattr(config, "COST_DISPLAY_CURRENCY", "USD")
            if currency == "INR":
                rate = getattr(config, "USD_TO_INR", 92.0)
                display_cost = total_cost_usd * rate
                cost_label = f"₹{display_cost:,.2f}"
            else:
                cost_label = f"${total_cost_usd:.4f}"
            col1, col2, col3 = st.columns(3)
            col1.metric("Input",  f"{total_in:,}")
            col2.metric("Output", f"{total_out:,}")
            col3.metric("API cost", cost_label)

        st.divider()

        # ── Re-initialise ────────────────────────────────────────────────────
        if st.button("Re-initialise pipeline", use_container_width=True):
            schema_source: object = config.SCHEMA_FILE_PATH or _default_schema()
            _build_pipeline(schema_source)
            st.rerun()

        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # ── Suggest business context ──────────────────────────────────────
        st.subheader("Suggest Business Context")
        st.caption(
            "Share domain knowledge (e.g. metric definitions, business rules) "
            "that helps the assistant answer better. An admin will review it."
        )
        ctx_text = st.text_area(
            "Your suggestion",
            key="ctx_suggestion_input",
            height=80,
            placeholder='e.g. "Free consultations" means order_category = \'Free\' …',
        )
        if st.button("Submit suggestion", use_container_width=True, key="submit_ctx"):
            if ctx_text and ctx_text.strip():
                add_suggestion(ctx_text.strip())
                st.success("Submitted! An admin will review your suggestion.")
            else:
                st.warning("Please enter some context before submitting.")

        approved = get_approved_context()
        if approved:
            with st.expander(f"Active business context ({len(approved)})", expanded=False):
                for entry in approved:
                    st.markdown(f"- {entry['text']}")

        # ── Conversation logs browser ─────────────────────────────────────
        if getattr(config, "CONVERSATION_LOG_ENABLED", False):
            st.divider()
            st.subheader("Conversation logs")
            log_base = getattr(
                config,
                "CONVERSATION_LOG_DIR",
                str(_ROOT / "logs" / "conversations"),
            )
            log_dir = Path(log_base)
            if log_dir.exists():
                log_files = sorted(log_dir.glob("*.json"), reverse=True)
                if log_files:
                    st.caption(f"{len(log_files)} conversation(s) logged")
                    selected_log = st.selectbox(
                        "Browse trace logs",
                        options=log_files,
                        format_func=lambda p: p.stem,
                        key="log_browser",
                    )
                    if selected_log and st.button("Load trace", key="load_trace"):
                        try:
                            trace = json.loads(selected_log.read_text(encoding="utf-8"))
                            st.session_state["_browsed_trace"] = trace
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
                else:
                    st.caption("No conversation logs yet.")
            else:
                st.caption("Log directory not created yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Chat message rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single chat message with optional debug panel."""
    with st.chat_message(msg["role"]):
        meta = msg.get("meta")
        # Show duration in seconds at the start for assistant answers (always visible, not just debug)
        if msg["role"] == "assistant" and meta:
            duration_ms = meta.get("end_to_end_ms") or meta.get("total_duration_ms")
            if duration_ms is not None and duration_ms >= 0:
                sec = duration_ms / 1000.0
                st.caption(f"This took **{sec:.1f}** seconds.")
        st.markdown(msg["content"])

        if not meta or msg["role"] != "assistant":
            return

        if st.session_state.get("show_debug"):
            with st.expander("Debug details", expanded=False):
                # ── Summary metrics ──────────────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows",       meta.get("rows_returned", "—"))
                col2.metric("Model",      meta.get("model_used",    "—"))
                col3.metric("Iterations", meta.get("query_iterations", "—"))
                col4.metric(
                    "Complexity",
                    meta.get("complexity", "—").upper()
                    if meta.get("complexity") else "—",
                )

                # End-to-end timing (UI perspective)
                end_to_end_ms = meta.get("end_to_end_ms")
                pipeline_ms = meta.get("total_duration_ms")
                if end_to_end_ms is not None:
                    tcol1, tcol2 = st.columns(2)
                    tcol1.metric("End-to-end (ms)", f"{end_to_end_ms:.0f}")
                    if pipeline_ms is not None:
                        tcol2.metric("Pipeline only (ms)", f"{pipeline_ms:.0f}")

                # ── High-level pipeline flow ─────────────────────────────
                st.subheader("Pipeline flow")
                complexity = meta.get("complexity", "—")
                iterations = meta.get("query_iterations", "—")
                st.markdown(
                    "- **1. Complexity estimation** → classify the question as "
                    f"**{complexity.upper() if isinstance(complexity, str) else complexity}** and "
                    "pick an initial model.\n"
                    "- **2. Query planning** → decide if multiple queries are needed and plan one "
                    "or more sub-queries with their target tables.\n"
                    f"- **3. Iterative loop** → for up to **{iterations}** iteration(s), do:\n"
                    "  - generate SQL with the LLM\n"
                    "  - validate the SQL against the known schema\n"
                    "  - execute the SQL on the database (with row caps)\n"
                    "  - optionally retry with a stronger model or different query if needed.\n"
                    "- **4. Result interpretation** → send the final SQL + result table back to the "
                    "LLM to generate the natural-language answer you see above."
                )

                # ── Final SQL ────────────────────────────────────────────
                if meta.get("sql_used"):
                    st.subheader("Final SQL")
                    st.code(meta["sql_used"], language="sql")

                # ── LLM prompts ──────────────────────────────────────────
                final_prompt_system = meta.get("interpretation_prompt_system") or ""
                final_prompt_messages = meta.get("interpretation_prompt_messages") or []
                if final_prompt_system or final_prompt_messages:
                    st.subheader("Final interpretation prompt")
                    if final_prompt_system:
                        st.markdown("**System prompt**")
                        st.code(final_prompt_system, language="markdown")
                    if final_prompt_messages:
                        st.markdown("**Messages**")
                        st.code(str(final_prompt_messages), language="json")

                # ── Per-step breakdown ───────────────────────────────────
                steps = meta.get("steps", [])
                if steps:
                    st.subheader("Query steps")

                    # Compact flow summary per iteration
                    for step in steps:
                        val_ok = not step.get("validation_errors")
                        exec_err = step.get("execution_error")
                        rows = step.get("rows_returned", 0)
                        summary_line = (
                            f"- **Iter {step['iteration']}** · model **{step['model_used']}** · "
                            f"confidence **{step['confidence']:.0%}** · "
                            f"validation: {'ok' if val_ok else 'errors'} · "
                            f"execution: {'error' if exec_err else 'ok'} · rows: {rows}"
                        )
                        st.markdown(summary_line)

                    # Detailed expandable view per step
                    for step in steps:
                        sub_desc = step.get("subquery_description") or ""
                        header_label = (
                            f"Step {step['iteration']} — {step['model_used']} "
                            f"(confidence: {step['confidence']:.0%})"
                        )
                        if sub_desc:
                            header_label = f"{header_label} · {sub_desc}"

                        with st.expander(
                            header_label,
                            expanded=False,
                        ):
                            # SQL
                            if step.get("sql_generated"):
                                st.markdown("**SQL**")
                                st.code(step["sql_generated"], language="sql")

                            # Prompt used for this step
                            if step.get("sql_prompt_system") or step.get("sql_prompt_messages"):
                                st.markdown("**Prompt**")
                                if step.get("sql_prompt_system"):
                                    st.caption("System")
                                    st.code(step["sql_prompt_system"], language="markdown")
                                if step.get("sql_prompt_messages"):
                                    st.caption("Messages")
                                    st.code(str(step["sql_prompt_messages"]), language="json")

                            # Reasoning and errors
                            if step.get("reasoning"):
                                st.markdown("**Reasoning**")
                                st.caption(step["reasoning"])
                            if step.get("validation_errors"):
                                st.markdown("**Validation errors**")
                                st.error(
                                    "; ".join(step["validation_errors"])
                                )
                            if step.get("execution_error"):
                                st.markdown("**Execution error**")
                                st.error(step["execution_error"])
                            if sub_desc:
                                st.caption(f"Sub-query: {sub_desc}")
                            st.caption(f"Rows returned: {step.get('rows_returned', 0)}")

                # ── Token summary and cost ─────────────────────────────────
                token_summary = meta.get("token_summary", {})
                if token_summary:
                    st.subheader("Token usage (this question)")
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Input",  f"{token_summary.get('input_tokens',  0):,}")
                    tc2.metric("Output", f"{token_summary.get('output_tokens', 0):,}")
                    tc3.metric("Calls",  token_summary.get("llm_calls", 0))
                    cost = token_summary.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        currency = getattr(config, "COST_DISPLAY_CURRENCY", "USD")
                        if currency == "INR":
                            rate = getattr(config, "USD_TO_INR", 92.0)
                            cost_label = f"₹{cost * rate:,.2f}"
                        else:
                            cost_label = f"${cost:.4f}"
                        tc4.metric("API cost", cost_label)
                    else:
                        tc4.metric("API cost", "—")

                st.caption(
                    f"Total duration: {meta.get('total_duration_ms', 0):.0f} ms"
                )

                # ── Conversation trace log ──────────────────────────────
                conv_id = meta.get("conversation_id")
                if conv_id:
                    st.divider()
                    st.subheader("Conversation trace")
                    st.code(f"ID: {conv_id}", language=None)

                    log_base = getattr(
                        config,
                        "CONVERSATION_LOG_DIR",
                        str(_ROOT / "logs" / "conversations"),
                    )
                    log_path = Path(log_base) / f"{conv_id}.json"
                    if log_path.exists():
                        with st.expander("View full conversation trace", expanded=False):
                            try:
                                trace_data = json.loads(log_path.read_text(encoding="utf-8"))

                                # Summary
                                st.markdown("**Trace summary**")
                                final = trace_data.get("final_result", {})
                                scol1, scol2, scol3 = st.columns(3)
                                scol1.metric("Stages logged", trace_data.get("stage_count", 0))
                                scol2.metric("Truncations", trace_data.get("truncation_count", 0))
                                scol3.metric("Duration (ms)", f"{final.get('total_duration_ms', 0):.0f}")

                                # Truncation alerts
                                truncations = trace_data.get("truncation_events", [])
                                if truncations:
                                    st.markdown("**Truncation events** (data was cut short here)")
                                    for trunc in truncations:
                                        st.warning(
                                            f"**{trunc['location']}**: "
                                            f"{trunc['original_size']:,} → {trunc['truncated_size']:,} {trunc['unit']}  \n"
                                            f"{trunc.get('detail', '')}"
                                        )

                                # Pipeline stages
                                st.markdown("**Pipeline stages**")
                                for stage_entry in trace_data.get("pipeline_stages", []):
                                    stage_name = stage_entry.get("stage", "unknown")
                                    with st.expander(f"📋 {stage_name}", expanded=False):
                                        st.json(stage_entry)

                            except Exception as trace_exc:
                                st.error(f"Could not load trace: {trace_exc}")
                    else:
                        st.caption(f"Trace file: `{log_path}`")

        # ── Error display ─────────────────────────────────────────────────
        if meta.get("error"):
            st.error(f"Pipeline error: {meta['error']}")


# ─────────────────────────────────────────────────────────────────────────────
# DeepAnalyze report section (local vLLM, no API)
# ─────────────────────────────────────────────────────────────────────────────

# Max rows/chars to send to DeepAnalyze to stay within context limits.
_DEEPANALYZE_MAX_ROWS = 500
_DEEPANALYZE_MAX_CHARS = 35_000


def _dataframe_to_report_data(df) -> str:
    """Convert DataFrame to CSV string for report prompt, truncated."""
    if df is None or df.empty:
        return "(no data)"
    subset = df.head(_DEEPANALYZE_MAX_ROWS)
    csv_str = subset.to_csv(index=False)
    if len(csv_str) > _DEEPANALYZE_MAX_CHARS:
        csv_str = csv_str[:_DEEPANALYZE_MAX_CHARS] + "\n... [truncated]"
    return csv_str


def _render_deepanalyze_report_section() -> None:
    """Render 'Generate report with DeepAnalyze' button and optional report/error."""
    if not config.DEEPANALYZE_ENABLED:
        return

    last_df = st.session_state.get("last_result_df")
    if last_df is not None and generate_report is not None:
        if st.button("Generate report with DeepAnalyze", type="secondary"):
            instruction = (
                "Generate a concise data science report. "
                f"User question: {st.session_state.get('last_result_question', '')}"
            )
            data_str = _dataframe_to_report_data(last_df)
            with st.spinner("Generating report…"):
                try:
                    report = generate_report(instruction, data_str)
                    st.session_state["last_deepanalyze_report"] = report
                    st.session_state["last_deepanalyze_error"] = None
                except DeepAnalyzeError as e:
                    st.session_state["last_deepanalyze_report"] = None
                    st.session_state["last_deepanalyze_error"] = str(e)
            st.rerun()

    if st.session_state.get("last_deepanalyze_report"):
        with st.expander("DeepAnalyze report", expanded=True):
            st.markdown(st.session_state["last_deepanalyze_report"])
    if st.session_state.get("last_deepanalyze_error"):
        st.error(st.session_state["last_deepanalyze_error"])

    if config.DEEPANALYZE_ENABLED and last_df is None and st.session_state.get("messages"):
        st.caption(
            "Run a question above to enable **Generate report with DeepAnalyze** "
            "(uses local vLLM; see docs/DEEPANALYZE_LOCAL.md)."
        )

    # Report from uploaded file (optional)
    if config.DEEPANALYZE_ENABLED and generate_report is not None:
        with st.expander("Report from uploaded file"):
            uploaded = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key="deepanalyze_file_upload",
            )
            if uploaded is not None and st.button("Generate report from file", key="deepanalyze_from_file"):
                with st.spinner("Reading file and generating report…"):
                    df_up = None
                    try:
                        if uploaded.name.lower().endswith(".csv"):
                            df_up = pd.read_csv(uploaded)
                        else:
                            try:
                                df_up = pd.read_excel(uploaded)
                            except Exception as exc:
                                st.error(
                                    "Excel support requires openpyxl. Install with: pip install openpyxl. "
                                    f"Error: {exc}"
                                )
                        if df_up is not None:
                            data_str = _dataframe_to_report_data(df_up)
                            instruction = "Generate a concise data science report for the following uploaded data."
                            report = generate_report(instruction, data_str)
                            st.session_state["last_deepanalyze_report"] = report
                            st.session_state["last_deepanalyze_error"] = None
                            st.rerun()
                    except DeepAnalyzeError as e:
                        st.session_state["last_deepanalyze_error"] = str(e)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to read file or generate report: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # Decide schema source: external YAML path from config if set, else default.
    schema_source: object = config.SCHEMA_FILE_PATH or _default_schema()

    # Lazy pipeline initialisation
    if st.session_state["orchestrator"] is None and st.session_state["init_error"] is None:
        with st.spinner("Initialising analytics pipeline…"):
            _build_pipeline(schema_source)

    _render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    cm = st.session_state.get("conn_manager")
    backend_label = f" · {cm.dialect_name}" if cm else ""
    st.caption(
        f"Ask questions about your data in plain English{backend_label}. "
        "The assistant will generate SQL, run it against your database, and explain the results."
    )

    if st.session_state["init_error"]:
        st.error(
            f"**Initialisation failed:** {st.session_state['init_error']}  \n"
            "Please check your `.env` configuration and click "
            "**Re-initialise pipeline** in the sidebar."
        )
        st.stop()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state["messages"]:
        _render_message(msg)

    # ── DeepAnalyze report (local vLLM) ────────────────────────────────────────
    _render_deepanalyze_report_section()

    # ── Browsed trace viewer ──────────────────────────────────────────────────
    browsed_trace = st.session_state.get("_browsed_trace")
    if browsed_trace:
        with st.expander(
            f"📜 Browsed trace: {browsed_trace.get('conversation_id', 'unknown')}",
            expanded=True,
        ):
            st.markdown(f"**Question:** {browsed_trace.get('question', '?')}")

            final = browsed_trace.get("final_result", {})
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Rows", final.get("rows_returned", 0))
            bcol2.metric("Iterations", final.get("query_iterations", 0))
            bcol3.metric("Truncations", browsed_trace.get("truncation_count", 0))
            bcol4.metric("Duration", f"{final.get('total_duration_ms', 0):.0f}ms")

            truncations = browsed_trace.get("truncation_events", [])
            if truncations:
                st.markdown("### Truncation events")
                for trunc in truncations:
                    st.warning(
                        f"**{trunc['location']}**: "
                        f"{trunc['original_size']:,} → {trunc['truncated_size']:,} {trunc['unit']}  \n"
                        f"{trunc.get('detail', '')}"
                    )

            st.markdown("### Pipeline stages")
            for stage_entry in browsed_trace.get("pipeline_stages", []):
                stage_name = stage_entry.get("stage", "unknown")
                with st.expander(f"📋 {stage_name}", expanded=False):
                    st.json(stage_entry)

            st.markdown("### Final answer")
            st.markdown(final.get("answer", "(no answer)"))

            if final.get("sql_used"):
                st.markdown("### SQL used")
                st.code(final["sql_used"], language="sql")

            if st.button("Close trace viewer", key="close_trace"):
                st.session_state["_browsed_trace"] = None
                st.rerun()

    # ── Input box ─────────────────────────────────────────────────────────────
    question = st.chat_input(
        placeholder="e.g. What were the top 5 products by revenue last month?",
        disabled=st.session_state["orchestrator"] is None,
    )

    if question:
        # Display user message immediately
        st.session_state["messages"].append({
            "role":    "user",
            "content": question,
        })
        _render_message(st.session_state["messages"][-1])

        # Build chat history for LLM context (last N turns, excluding current one)
        history_depth = config.CHAT_CONTEXT_TURNS
        prior_messages = st.session_state["messages"][:-1]
        if history_depth > 0 and prior_messages:
            max_msgs = history_depth * 2  # approx user+assistant per turn
            chat_history = prior_messages[-max_msgs:]
        else:
            chat_history = []

        # Run the pipeline
        with st.chat_message("assistant"):
            # Live status feed — always visible, last 3 steps in small font
            _status_placeholder = st.empty()
            _status_history: list[str] = []

            def _status_callback(msg: str) -> None:
                """Push a status message and render the most recent 3, one per line."""
                _status_history.append(msg)
                recent = _status_history[-3:]
                blocks = [
                    f'<div style="font-size:0.82em;line-height:1.6;color:#888;margin:2px 0">'
                    f"{'⏳ <strong>' if i == len(recent) - 1 else '✅ '}"
                    f"{s}{'</strong>' if i == len(recent) - 1 else ''}"
                    f"</div>"
                    for i, s in enumerate(recent)
                ]
                _status_placeholder.markdown(
                    "".join(blocks),
                    unsafe_allow_html=True,
                )

            live_debug_placeholder = (
                st.empty() if st.session_state.get("show_debug") else None
            )

            def _progress_callback(step) -> None:
                """Stream live debug updates for each query iteration."""
                if live_debug_placeholder is None:
                    return
                with live_debug_placeholder.container():
                    sub_desc = getattr(step, "subquery_description", "") or ""
                    header = (
                        f"Live debug — step {step.iteration} "
                        f"(model: {step.model_used})"
                    )
                    if sub_desc:
                        header = f"{header} · {sub_desc}"
                    st.subheader(header)
                    if step.sql_generated:
                        st.markdown("**SQL candidate**")
                        st.code(step.sql_generated, language="sql")
                    if step.validation_errors:
                        st.markdown("**Validation errors**")
                        st.error("; ".join(step.validation_errors))
                    if step.execution_error:
                        st.markdown("**Execution error**")
                        st.error(step.execution_error)
                    else:
                        st.markdown(f"**Rows returned**: {step.rows_returned}")

            try:
                orchestrator = st.session_state["orchestrator"]
                # Measure end-to-end time from question to final result.
                with Timer() as end_to_end_timer:
                    try:
                        result = orchestrator.run(
                            question,
                            chat_history=chat_history,
                            progress_callback=(
                                _progress_callback if live_debug_placeholder else None
                            ),
                            status_callback=_status_callback,
                        )
                    except TypeError as exc:
                        # Backwards compatibility for older orchestrator instances
                        if "status_callback" in str(exc) or "progress_callback" in str(exc):
                            result = orchestrator.run(
                                question,
                                chat_history=chat_history,
                            )
                        else:
                            raise

                # Clear the live status feed once the answer is ready
                _status_placeholder.empty()

                # Build meta dict for the rendered message
                meta = {
                    "sql_used":         result.sql_used,
                    "rows_returned":    result.rows_returned,
                    "model_used":       result.model_used,
                    "complexity":       result.complexity.value,
                    "query_iterations": result.query_iterations,
                    "end_to_end_ms":    getattr(end_to_end_timer, "elapsed_ms", result.total_duration_ms),
                    "conversation_id":  getattr(result, "conversation_id", None),
                    "interpretation_prompt_system":   getattr(result, "interpretation_prompt_system", None),
                    "interpretation_prompt_messages": getattr(result, "interpretation_prompt_messages", None),
                    "steps":            [
                        {
                            "iteration":            s.iteration,
                            "model_used":           s.model_used,
                            "sql_generated":        s.sql_generated,
                            "confidence":           s.confidence,
                            "reasoning":            s.reasoning,
                            "validation_errors":    s.validation_errors,
                            "execution_error":      s.execution_error,
                            "rows_returned":        s.rows_returned,
                            "subquery_id":          getattr(s, "subquery_id", None),
                            "subquery_description": getattr(s, "subquery_description", ""),
                            "sql_prompt_system":    getattr(s, "sql_prompt_system", ""),
                            "sql_prompt_messages":  getattr(s, "sql_prompt_messages", []),
                        }
                        for s in result.steps
                    ],
                    "total_duration_ms": result.total_duration_ms,
                    "token_summary":     result.token_summary,
                    "error":             result.error,
                }

                assistant_msg = {
                    "role":    "assistant",
                    "content": result.answer,
                    "meta":    meta,
                }
                st.session_state["messages"].append(assistant_msg)

                # Store last result for DeepAnalyze report generation
                if result.error is None and getattr(result, "result_dataframe", None) is not None:
                    st.session_state["last_result_df"] = result.result_dataframe
                    st.session_state["last_result_question"] = question
                    st.session_state["last_result_sql"] = result.sql_used
                    st.session_state["last_deepanalyze_error"] = None

            except Exception as exc:  # noqa: BLE001
                _status_placeholder.empty()
                err_msg = f"Unexpected error: {exc}"
                log.error("ui_pipeline_error", error=str(exc), traceback=traceback.format_exc())
                assistant_msg = {
                    "role":    "assistant",
                    "content": "An unexpected error occurred. Please check the logs.",
                    "meta":    {"error": err_msg},
                }
                st.session_state["messages"].append(assistant_msg)

        # Rerun to re-render history with the new messages (including debug panels)
        st.rerun()


if __name__ == "__main__":
    main()
