"""
LLM-powered chart builder.

Instead of heuristic axis/type inference, this module sends a sample of the
query result data to the LLM and asks it to write Plotly code that best
visualises the data for the user's question.  The generated code is executed
in a sandboxed namespace and the resulting ``plotly.graph_objects.Figure`` is
serialised to a dict for the Streamlit UI.
"""
from __future__ import annotations

import textwrap
import traceback
from typing import Any

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Max rows / chars of data we send to the LLM (keep token usage small)
_MAX_SAMPLE_ROWS = 30
_MAX_DATA_CHARS = 4_000

_CHART_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert data visualisation engineer. Given a pandas DataFrame and the
user's analytics question, write Python code that creates the single best
Plotly chart to answer the question visually.

## Rules
1. Use ONLY `plotly.graph_objects` (imported as `go`) — do NOT use plotly.express.
2. The code must define a variable called `fig` that is a `go.Figure`.
3. You receive the full DataFrame as the variable `df` (a pandas DataFrame).
4. Choose the chart type that best fits the data and question:
   - Time series / trends → line chart (with markers)
   - Comparisons / breakdowns → bar chart (horizontal if many categories)
   - Proportions → pie / donut chart
   - Distributions → histogram or box plot
   - Correlations → scatter plot
5. Make the chart beautiful and readable:
   - Add a clear, concise title derived from the question.
   - Label axes with human-readable names (replace underscores with spaces, title-case).
   - Use a clean colour palette.
   - Format large numbers with comma separators on axes.
   - Rotate x-axis labels if they are long dates/text.
   - Add hover info that shows exact values.
6. For time-series data: sort by the date column, format dates nicely on x-axis.
7. If there are multiple numeric series worth showing, use multiple traces.
8. Keep the layout clean: `margin=dict(l=60, r=40, t=60, b=80)`, `height=450`.
9. Do NOT call `fig.show()` or `fig.write_*()`. Just create `fig`.
10. Do NOT import anything other than `plotly.graph_objects as go`. The `df` and
    `go` variables are pre-injected — do NOT re-import pandas or re-create df.
11. Output ONLY the Python code. No explanation, no markdown fences, no comments
    outside the code.
""")


def _build_chart_prompt(
    df: pd.DataFrame,
    question: str,
) -> str:
    """Build the user message with schema info and sample data."""
    sample = df.head(_MAX_SAMPLE_ROWS)
    csv_str = sample.to_csv(index=False)
    if len(csv_str) > _MAX_DATA_CHARS:
        csv_str = csv_str[:_MAX_DATA_CHARS] + "\n... [truncated]"

    dtypes_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())

    return textwrap.dedent(f"""\
## User question
{question}

## DataFrame info
Shape: {df.shape[0]} rows x {df.shape[1]} columns

Column dtypes:
{dtypes_str}

## Sample data (CSV)
{csv_str}

Write the Plotly code now. Output ONLY Python code, nothing else.
""")


def _extract_code(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Drop opening fence (```python or ```)
        lines = lines[1:]
        # Drop closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def _execute_chart_code(code: str, df: pd.DataFrame) -> "go.Figure | None":
    """Execute LLM-generated Plotly code in a restricted namespace."""
    import plotly.graph_objects as go

    namespace: dict[str, Any] = {
        "go": go,
        "df": df.copy(),
        "pd": pd,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        log.warning("chart_code_exec_failed", error=traceback.format_exc())
        return None

    fig = namespace.get("fig")
    if fig is None or not isinstance(fig, go.Figure):
        log.warning("chart_code_no_fig", keys=list(namespace.keys()))
        return None

    return fig


def build_chart(
    df: pd.DataFrame,
    question_hint: str = "",
    llm_client: Any = None,
    model: str | None = None,
) -> dict | None:
    """
    Build a Plotly chart by asking the LLM to write the visualisation code.

    Parameters
    ----------
    df:             Query result DataFrame.
    question_hint:  The user's original question (helps the LLM pick chart type).
    llm_client:     LLMClient instance. If None, falls back to a simple heuristic.
    model:          Model to use for code generation (defaults to LOW_MODEL).

    Returns
    -------
    Plotly figure as a dict (serialisable), or None if chart building fails.
    """
    if df is None or df.empty or len(df.columns) < 2:
        log.info("chart_skip", reason="insufficient_data", shape=df.shape if df is not None else None)
        return None

    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        log.warning(
            "chart_skip",
            reason="plotly_not_installed",
            error=str(e),
            hint="Run: pip install plotly",
        )
        return None

    # Coerce string columns that are actually numeric
    df = df.head(500).copy()
    for c in df.columns:
        if df[c].dtype == object:
            try:
                numeric = pd.to_numeric(df[c], errors="coerce")
                if numeric.notna().sum() > len(df) * 0.5:
                    df[c] = numeric
            except Exception:  # noqa: BLE001
                pass

    # If no LLM client, fall back to basic heuristic
    if llm_client is None:
        log.info("chart_fallback_heuristic", reason="no_llm_client")
        return _fallback_heuristic(df, question_hint)

    from config import config as cfg
    chart_model = model or cfg.LOW_MODEL

    prompt = _build_chart_prompt(df, question_hint)

    try:
        response = llm_client.complete(
            model=chart_model,
            messages=[{"role": "user", "content": prompt}],
            system=_CHART_SYSTEM_PROMPT,
            max_tokens=2_048,
            temperature=0.0,
        )
        raw_code = response.content
        log.info(
            "chart_llm_response",
            model=chart_model,
            code_length=len(raw_code),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("chart_llm_call_failed", error=str(e))
        return _fallback_heuristic(df, question_hint)

    code = _extract_code(raw_code)
    if not code:
        log.warning("chart_empty_code")
        return _fallback_heuristic(df, question_hint)

    log.info("chart_executing_code", code_preview=code[:300])

    fig = _execute_chart_code(code, df)
    if fig is None:
        log.warning("chart_exec_returned_none", falling_back="heuristic")
        return _fallback_heuristic(df, question_hint)

    try:
        fig_dict = fig.to_dict()
        log.info("chart_built_via_llm", traces=len(fig_dict.get("data", [])))
        return fig_dict
    except Exception as e:  # noqa: BLE001
        log.warning("chart_serialization_failed", error=str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fallback heuristic (used when no LLM client available)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_heuristic(df: pd.DataFrame, question_hint: str) -> dict | None:
    """Simple dtype-based chart builder as a safety net."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    cols = list(df.columns)
    date_cols = [
        c for c in cols
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or (df[c].dtype == object and _looks_like_date(df[c]))
    ]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in cols if c not in numeric_cols and c not in date_cols]

    x_col: str | None = None
    y_cols: list[str] = []
    chart_type = "bar"

    if date_cols and numeric_cols:
        x_col = date_cols[0]
        y_cols = numeric_cols[:3]
        chart_type = "line"
    elif other_cols and numeric_cols:
        x_col = other_cols[0]
        y_cols = numeric_cols[:3]
    elif len(cols) >= 2:
        x_col = cols[0]
        y_cols = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])][:3]

    if not x_col or not y_cols:
        return None

    fig = go.Figure()
    for y in y_cols:
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=df[x_col].astype(str).tolist(),
                y=df[y].tolist(),
                mode="lines+markers",
                name=y.replace("_", " ").title(),
            ))
        else:
            fig.add_trace(go.Bar(
                x=df[x_col].astype(str).tolist(),
                y=df[y].tolist(),
                name=y.replace("_", " ").title(),
            ))

    fig.update_layout(
        margin=dict(l=60, r=40, t=50, b=80),
        height=450,
        showlegend=len(y_cols) > 1,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if chart_type == "bar" and len(y_cols) > 1:
        fig.update_layout(barmode="group")

    log.info("chart_built_via_heuristic", chart_type=chart_type, x=x_col, y=y_cols)
    return fig.to_dict()


def _looks_like_date(series: pd.Series) -> bool:
    """Heuristic: series of strings that look like dates."""
    try:
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False
        converted = pd.to_datetime(sample, errors="coerce")
        return converted.notna().sum() > len(sample) * 0.5
    except Exception:  # noqa: BLE001
        return False
