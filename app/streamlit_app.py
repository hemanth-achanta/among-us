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
from utils.logger import configure_logging, get_logger

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

        # ── Token usage ──────────────────────────────────────────────────────
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
            col1, col2 = st.columns(2)
            col1.metric("Input",  f"{total_in:,}")
            col2.metric("Output", f"{total_out:,}")

        st.divider()

        # ── Re-initialise ────────────────────────────────────────────────────
        if st.button("Re-initialise pipeline", use_container_width=True):
            schema_source: object = config.SCHEMA_FILE_PATH or _default_schema()
            _build_pipeline(schema_source)
            st.rerun()

        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat message rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single chat message with optional debug panel."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        meta = msg.get("meta")
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

                # ── Token summary ────────────────────────────────────────
                token_summary = meta.get("token_summary", {})
                if token_summary:
                    st.subheader("Token usage (this question)")
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Input",  f"{token_summary.get('input_tokens',  0):,}")
                    tc2.metric("Output", f"{token_summary.get('output_tokens', 0):,}")
                    tc3.metric("Calls",  token_summary.get("llm_calls", 0))

                st.caption(
                    f"Total duration: {meta.get('total_duration_ms', 0):.0f} ms"
                )

        # ── Error display ─────────────────────────────────────────────────
        if meta.get("error"):
            st.error(f"Pipeline error: {meta['error']}")


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
            with st.spinner("Analysing your question…"):
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
                    try:
                        # Preferred path: orchestrator that supports live progress.
                        result = orchestrator.run(
                            question,
                            chat_history=chat_history,
                            progress_callback=(
                                _progress_callback if live_debug_placeholder else None
                            ),
                        )
                    except TypeError as exc:
                        # Backwards compatibility for older orchestrator instances
                        # that do not accept a progress_callback argument.
                        if "progress_callback" in str(exc):
                            result = orchestrator.run(
                                question,
                                chat_history=chat_history,
                            )
                        else:
                            raise

                    # Build meta dict for the rendered message
                    meta = {
                        "sql_used":         result.sql_used,
                        "rows_returned":    result.rows_returned,
                        "model_used":       result.model_used,
                        "complexity":       result.complexity.value,
                        "query_iterations": result.query_iterations,
                        # Use getattr for backwards compatibility while the app reloads.
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
                                # getattr for backwards compatibility with older QueryStep objects
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

                except Exception as exc:  # noqa: BLE001
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
