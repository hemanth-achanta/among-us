"""
Central configuration for the analytics assistant.
All secrets are loaded from .env; all tunables are defined here.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ── Enumerations ──────────────────────────────────────────────────────────────

class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL      = "mysql"
    SNOWFLAKE  = "snowflake"
    TRINO      = "trino"      # Presto/Trino via prestodb


class ComplexityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_TYPE: DatabaseType = DatabaseType(
    os.getenv("DATABASE_TYPE", DatabaseType.TRINO.value)
)

DB_HOST:     str = os.getenv("DB_HOST", "metabase-trino-master.prod.dataplatform.link")
DB_PORT:     int = int(os.getenv("DB_PORT", "8443"))
DB_USER:     str = os.getenv("DB_USER", "hemanth.achanta")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "Hey@12345")
DB_NAME:     str = os.getenv("DB_NAME", "lakehouse")   # Trino catalog
DB_SCHEMA:   str = os.getenv("DB_SCHEMA", "data_model")          # set per query if needed

# Snowflake-specific extras
SNOWFLAKE_ACCOUNT:    str = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_WAREHOUSE:  str = os.getenv("SNOWFLAKE_WAREHOUSE", "")
SNOWFLAKE_ROLE:       str = os.getenv("SNOWFLAKE_ROLE", "")

# Trino/Presto-specific extras
TRINO_HTTP_SCHEME:   str  = os.getenv("TRINO_HTTP_SCHEME",  "https")
TRINO_VERIFY_SSL:    bool = os.getenv("TRINO_VERIFY_SSL", "false").lower() == "true"
TRINO_CATALOG:       str  = os.getenv("TRINO_CATALOG", "hive")
TRINO_QUERY_TIMEOUT: int  = int(os.getenv("TRINO_QUERY_TIMEOUT", "120"))  # seconds

# SQLAlchemy pool settings
DB_POOL_SIZE:     int = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW:  int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT:  int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE:  int = int(os.getenv("DB_POOL_RECYCLE", "1800"))

# Max rows returned from any single query
MAX_RESULT_ROWS: int = int(os.getenv("MAX_RESULT_ROWS", "500"))


# ── LLM / API ─────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Model assignments per complexity tier
LOW_MODEL:    str = os.getenv("LOW_MODEL",    "claude-haiku-4-5-20251001")
MEDIUM_MODEL: str = os.getenv("MEDIUM_MODEL", "claude-sonnet-4-6")
HIGH_MODEL:   str = os.getenv("HIGH_MODEL",   "claude-sonnet-4-6")

# Escalation chain used by RetryManager (ordered weakest → strongest)
MODEL_ESCALATION_CHAIN: list[str] = [
    LOW_MODEL,
    MEDIUM_MODEL,
    HIGH_MODEL,
]

# Token budgets (approximate; enforced via context truncation in prompts)
TOKEN_LIMITS: dict[str, int] = {
    LOW_MODEL:    4_096,
    MEDIUM_MODEL: 8_192,
    HIGH_MODEL:   16_384,
}

# API cost: USD per 1M input tokens, USD per 1M output tokens (for cost display).
# Source: https://www.anthropic.com/pricing and https://docs.anthropic.com/en/docs/about-claude/pricing
# Keys are model identifiers; unknown models fall back to Sonnet pricing.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Claude 3.5 Haiku (Oct 2024)
    "claude-3-5-haiku-20241022":        (0.80, 4.00),
    # Claude Haiku 4.5
    "claude-haiku-4-5-20251001":        (1.00, 5.00),
    # Claude 3.5 Sonnet (Oct 2024)
    "claude-3-5-sonnet-20241022":       (3.00, 15.00),
    # Claude Sonnet 4.5 / 4.6
    "claude-sonnet-4-6":                (3.00, 15.00),
    "claude-sonnet-4-5-20250514":       (3.00, 15.00),
    # Claude Opus 4.6
    "claude-opus-4-6":                  (5.00, 25.00),
}

# Cost display: which currency to show in the UI. Set to "INR" or "USD" (env: COST_DISPLAY_CURRENCY).
_cost_currency = os.getenv("COST_DISPLAY_CURRENCY", "INR").upper()
COST_DISPLAY_CURRENCY: str = "INR" if _cost_currency == "INR" else "USD"
# USD→INR rate used when COST_DISPLAY_CURRENCY is INR (env: USD_TO_INR; update periodically).
USD_TO_INR: float = float(os.getenv("USD_TO_INR", "92.0"))

# Max tokens included for schema context in a single prompt.
# Must be large enough to fit all tables + relationships + join notes.
SCHEMA_CONTEXT_TOKEN_LIMIT: int = int(os.getenv("SCHEMA_CONTEXT_TOKEN_LIMIT", "8000"))

# Max sample rows injected per table in prompts (2 balances cost vs. context)
SCHEMA_SAMPLE_ROWS: int = int(os.getenv("SCHEMA_SAMPLE_ROWS", "2"))

# LLM temperature (0 = deterministic; keep low for SQL generation)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Hard timeout (seconds) for a single LLM call
LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))


# ── Orchestrator ──────────────────────────────────────────────────────────────

# Max independent SQL queries the orchestrator may run per user question
MAX_QUERY_ITERATIONS: int = int(os.getenv("MAX_QUERY_ITERATIONS", "5"))

# Max planned queries (sub-queries) per user question.
# This caps how many distinct query intents the planner may schedule; retries
# within a single sub-query are governed separately by MAX_RETRIES.
MAX_QUERIES_PER_QUESTION: int = int(
    os.getenv("MAX_QUERIES_PER_QUESTION", "4")
)

# Max retry attempts per SQL generation / execution cycle
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# Retry escalation threshold: upgrade model after this many failures
RETRY_ESCALATION_THRESHOLD: int = int(os.getenv("RETRY_ESCALATION_THRESHOLD", "2"))

# Minimum confidence score from SQL generator to skip retry
MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6"))

# If True, empty result sets trigger a retry
RETRY_ON_EMPTY_RESULT: bool = os.getenv("RETRY_ON_EMPTY_RESULT", "true").lower() == "true"


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL:  str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")   # "json" | "text"
LOG_FILE:   str = os.getenv("LOG_FILE", str(_ROOT / "logs" / "analytics.log"))

# Per-conversation trace logs (one JSON file per question)
CONVERSATION_LOG_ENABLED: bool = os.getenv("CONVERSATION_LOG_ENABLED", "true").lower() == "true"
CONVERSATION_LOG_DIR: str = os.getenv(
    "CONVERSATION_LOG_DIR",
    str(_ROOT / "logs" / "conversations"),
)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

APP_TITLE:       str  = "Analytics Assistant"
APP_ICON:        str  = "📊"
SHOW_DEBUG_INFO: bool = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"

# Optional path to a schema YAML file used by the Streamlit app.
# Defaults to ``<project_root>/schema.yaml`` so it works regardless of
# the current working directory when running Streamlit.
SCHEMA_FILE_PATH: str = os.getenv(
    "SCHEMA_FILE_PATH",
    str(_ROOT / "schema.yaml"),
)

# How many previous chat turns to include as context when building LLM prompts.
# This controls the depth of conversational memory per session.
CHAT_CONTEXT_TURNS: int = int(os.getenv("CHAT_CONTEXT_TURNS", "3"))


# ── DeepAnalyze (local vLLM report generation) ─────────────────────────────────

# When True, "Generate report with DeepAnalyze" is available (requires local vLLM).
DEEPANALYZE_ENABLED: bool = os.getenv("DEEPANALYZE_ENABLED", "false").lower() == "true"

# Base URL of local vLLM OpenAI-compatible API (no DeepAnalyze API server).
DEEPANALYZE_BASE_URL: str = os.getenv(
    "DEEPANALYZE_BASE_URL",
    "http://localhost:8000/v1",
)

# Model name as served by vLLM (must match vLLM's served name, usually the HF repo ID).
DEEPANALYZE_MODEL: str = os.getenv("DEEPANALYZE_MODEL", "RUC-DataLab/DeepAnalyze-8B")

# Timeout in seconds for report-generation requests.
DEEPANALYZE_REQUEST_TIMEOUT: int = int(
    os.getenv("DEEPANALYZE_REQUEST_TIMEOUT", "120"),
)


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Raise RuntimeError if critical configuration is missing.
    Call this once at application startup.
    """
    errors: list[str] = []

    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is not set.")
    if not DB_USER:
        errors.append("DB_USER is not set.")
    if not DB_PASSWORD:
        errors.append("DB_PASSWORD is not set.")

    # Trino only needs host+user+password; catalog is optional (defaults to lakehouse)
    if DATABASE_TYPE != DatabaseType.TRINO and not DB_NAME:
        errors.append("DB_NAME is not set.")

    if DATABASE_TYPE == DatabaseType.TRINO and not DB_HOST:
        errors.append("DB_HOST (Trino host) is not set.")

    if errors:
        raise RuntimeError(
            "Configuration errors detected:\n" + "\n".join(f"  • {e}" for e in errors)
        )
