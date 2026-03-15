"""
DeepAnalyze local report generation (Option A).

Uses the OpenAI-compatible HTTP API exposed by a local vLLM server
serving DeepAnalyze-8B. No DeepAnalyze API server (port 8200) or
external API is used; all traffic goes to the local vLLM endpoint.
"""
from __future__ import annotations

from openai import OpenAI

from config import config
from utils.logger import get_logger

log = get_logger(__name__)


class DeepAnalyzeError(Exception):
    """Raised when report generation fails (e.g. vLLM unreachable or timeout)."""


def generate_report(
    instruction: str,
    data_summary_or_csv: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """
    Ask the local DeepAnalyze model (via vLLM) to generate a data science report.

    Parameters
    ----------
    instruction : str
        User-facing instruction, e.g. the original question or "Generate a concise
        data science report. User question: …"
    data_summary_or_csv : str
        The data as CSV text or a short summary (columns/rows). Truncate if large
        to stay within context limits.
    base_url : str, optional
        Override config. Defaults to config.DEEPANALYZE_BASE_URL.
    model : str, optional
        Override config. Defaults to config.DEEPANALYZE_MODEL.
    timeout : int, optional
        Request timeout in seconds. Defaults to config.DEEPANALYZE_REQUEST_TIMEOUT.

    Returns
    -------
    str
        The generated report text.

    Raises
    ------
    DeepAnalyzeError
        If the client is disabled, or the request fails (connection, timeout, API error).
    """
    if not config.DEEPANALYZE_ENABLED:
        raise DeepAnalyzeError(
            "DeepAnalyze report generation is disabled. Set DEEPANALYZE_ENABLED=true "
            "and ensure a local vLLM server is running (see docs/DEEPANALYZE_LOCAL.md)."
        )

    _base_url = base_url or config.DEEPANALYZE_BASE_URL
    _model = model or config.DEEPANALYZE_MODEL
    _timeout = timeout if timeout is not None else config.DEEPANALYZE_REQUEST_TIMEOUT

    prompt = (
        f"{instruction}\n\n"
        "## Data\n\n"
        f"{data_summary_or_csv}"
    )

    # vLLM OpenAI-compatible server typically accepts any api_key.
    client = OpenAI(
        base_url=_base_url,
        api_key="dummy",
        timeout=float(_timeout),
    )

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
    except Exception as exc:
        log.warning(
            "deepanalyze_report_failed",
            base_url=_base_url,
            error=str(exc),
        )
        raise DeepAnalyzeError(
            f"Report generation failed: {exc}. Ensure vLLM is running at {_base_url} "
            "(see docs/DEEPANALYZE_LOCAL.md)."
        ) from exc

    choice = response.choices[0] if response.choices else None
    if not choice or not getattr(choice, "message", None):
        raise DeepAnalyzeError("Report generation returned an empty response.")

    content = choice.message.content or ""
    log.info(
        "deepanalyze_report_complete",
        base_url=_base_url,
        output_length=len(content),
    )
    return content.strip()
