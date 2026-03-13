"""
Database connection management.

Two managers are provided:

``ConnectionManager``
    SQLAlchemy-based engine for PostgreSQL, MySQL, and Snowflake.
    Uses a ``QueuePool`` for connection reuse.

``PrestoConnectionManager``
    Direct ``prestodb`` client for Trino / Presto endpoints.
    Mirrors the exact pattern from the project notebook:
        * HTTPS scheme
        * BasicAuthentication
        * SSL verification disabled (internal cluster)
    Exposes the same public interface as ``ConnectionManager`` so the rest of
    the codebase (orchestrator, executor) can use either transparently.

The factory function ``make_connection_manager()`` reads ``config.DATABASE_TYPE``
and returns the appropriate instance.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd

from config import config
from config.config import DatabaseType
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions (shared by both managers)
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseConnectionError(Exception):
    """Raised when a connection cannot be established."""


class QueryExecutionError(Exception):
    """Raised when a SQL statement fails at the database level."""


# ─────────────────────────────────────────────────────────────────────────────
# PrestoConnectionManager  (Trino / Presto via prestodb)
# ─────────────────────────────────────────────────────────────────────────────

class PrestoConnectionManager:
    """
    Manages connections to a Trino/Presto cluster using the ``prestodb`` library.

    Connection parameters mirror those in the project notebook exactly:
      - HTTPS scheme
      - BasicAuthentication
      - SSL verification disabled (self-signed / internal cert)

    Parameters
    ----------
    host:       Trino master hostname (e.g. ``metabase-trino-master.prod...``).
    port:       HTTPS port (typically 8443).
    user:       Trino username.
    password:   Trino password.
    catalog:    Default Trino catalog (e.g. ``lakehouse``).
    http_scheme: ``"https"`` (default) or ``"http"``.
    verify_ssl:  Whether to verify the TLS certificate (default ``False`` for
                 internal clusters with self-signed certs).
    """

    def __init__(
        self,
        host:        str  = config.DB_HOST,
        port:        int  = config.DB_PORT,
        user:        str  = config.DB_USER,
        password:    str  = config.DB_PASSWORD,
        catalog:     str  = config.TRINO_CATALOG,
        http_scheme: str  = config.TRINO_HTTP_SCHEME,
        verify_ssl:  bool = config.TRINO_VERIFY_SSL,
    ) -> None:
        self._host        = host
        self._port        = port
        self._user        = user
        self._password    = password
        self._catalog     = catalog
        self._http_scheme = http_scheme
        self._verify_ssl  = verify_ssl

        log.info(
            "presto_connection_manager_init",
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            http_scheme=http_scheme,
            verify_ssl=verify_ssl,
        )

    # ── Internal: raw prestodb connection ─────────────────────────────────────

    def _open_connection(self):
        """
        Open and return a new ``prestodb.dbapi.Connection``.

        Matches the notebook pattern exactly — creates a fresh connection per
        call. ``prestodb`` connections are lightweight; pooling is not required
        for read-only analytics workloads.
        """
        try:
            import prestodb
            import prestodb.auth
        except ImportError as exc:
            raise DatabaseConnectionError(
                "prestodb is not installed. Run: pip install presto-python-client"
            ) from exc

        try:
            conn = prestodb.dbapi.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                http_scheme=self._http_scheme,
                catalog=self._catalog,
                # Do not set a default schema so that queries use the fully-
                # qualified names from schema.yaml, e.g. ``hive.data_model.table``.
                schema=None,
                auth=prestodb.auth.BasicAuthentication(self._user, self._password),
            )
            # Disable SSL cert verification for internal clusters (notebook pattern)
            conn._http_session.verify = self._verify_ssl
            return conn
        except Exception as exc:
            log.error("presto_connection_failed", error=str(exc))
            raise DatabaseConnectionError(
                f"Could not connect to Trino at {self._host}:{self._port} — {exc}"
            ) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute_query(
        self,
        sql: str,
        max_rows: int = config.MAX_RESULT_ROWS,
    ) -> tuple[pd.DataFrame, bool]:
        """
        Execute *sql* and return ``(DataFrame, truncated)``.

        Parameters
        ----------
        sql:      SQL query to execute.
        max_rows: Maximum rows to fetch.

        Returns
        -------
        ``(dataframe, truncated)`` — *truncated* is ``True`` when the result
        was capped at *max_rows*.

        Raises
        ------
        QueryExecutionError: On any database-level error.
        """
        log.info("presto_execute_start", sql_snippet=sql[:200])

        # Suppress InsecureRequestWarning (same situation as notebook)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conn = self._open_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(sql)

                # Fetch max_rows + 1 to detect truncation
                rows = cursor.fetchmany(max_rows + 1)
                col_names = [desc[0] for desc in cursor.description or []]
            except Exception as exc:
                log.error("presto_execute_failed", error=str(exc))
                raise QueryExecutionError(str(exc)) from exc
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

        truncated    = len(rows) > max_rows
        limited_rows = rows[:max_rows]
        df           = pd.DataFrame(limited_rows, columns=col_names)

        log.info(
            "presto_execute_complete",
            row_count=len(df),
            col_count=len(col_names),
            truncated=truncated,
        )
        return df, truncated

    def health_check(self) -> bool:
        """Verify connectivity with a trivial query."""
        try:
            df, _ = self.execute_query("SELECT 1 AS ping", max_rows=1)
            ok = not df.empty
            log.info("presto_health_check_passed" if ok else "presto_health_check_empty")
            return ok
        except Exception as exc:  # noqa: BLE001
            log.error("presto_health_check_failed", error=str(exc))
            return False

    def dispose(self) -> None:
        """No-op for API compatibility — connections are per-query."""
        log.debug("presto_dispose_noop")

    @property
    def dialect_name(self) -> str:
        return "Presto SQL (Trino)"

    @property
    def catalog(self) -> str:
        return self._catalog


# ─────────────────────────────────────────────────────────────────────────────
# ConnectionManager  (SQLAlchemy — PostgreSQL / MySQL / Snowflake)
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Manages a single shared SQLAlchemy engine for relational databases.

    Supports PostgreSQL, MySQL, and Snowflake via QueuePool.
    Use ``PrestoConnectionManager`` for Trino/Presto endpoints.
    """

    def __init__(self) -> None:
        import sqlalchemy as sa  # lazy import — not needed for Trino path
        from sqlalchemy.pool import QueuePool  # noqa: F401
        self._sa = sa
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Any:
        import sqlalchemy as sa
        from sqlalchemy.pool import QueuePool

        db_type = config.DATABASE_TYPE
        url = self._build_url(db_type)

        connect_args: dict = {}
        if db_type in (DatabaseType.POSTGRESQL, DatabaseType.MYSQL):
            connect_args["connect_timeout"] = 10

        engine = sa.create_engine(
            url,
            poolclass=QueuePool,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            pool_timeout=config.DB_POOL_TIMEOUT,
            pool_recycle=config.DB_POOL_RECYCLE,
            pool_pre_ping=True,
            connect_args=connect_args,
            echo=False,
        )
        log.info("sqlalchemy_engine_created", database_type=db_type.value)
        return engine

    @staticmethod
    def _build_url(db_type: DatabaseType) -> str:
        if db_type == DatabaseType.POSTGRESQL:
            return (
                f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASSWORD}"
                f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
            )
        if db_type == DatabaseType.MYSQL:
            return (
                f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}"
                f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
            )
        if db_type == DatabaseType.SNOWFLAKE:
            base = (
                f"snowflake://{config.DB_USER}:{config.DB_PASSWORD}"
                f"@{config.SNOWFLAKE_ACCOUNT}/{config.DB_NAME}/{config.DB_SCHEMA}"
            )
            parts: list[str] = []
            if config.SNOWFLAKE_WAREHOUSE:
                parts.append(f"warehouse={config.SNOWFLAKE_WAREHOUSE}")
            if config.SNOWFLAKE_ROLE:
                parts.append(f"role={config.SNOWFLAKE_ROLE}")
            if parts:
                base += "?" + "&".join(parts)
            return base
        raise DatabaseConnectionError(
            f"Unsupported DATABASE_TYPE for SQLAlchemy engine: '{db_type}'. "
            "Use PrestoConnectionManager for Trino."
        )

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Yield an active SQLAlchemy Connection."""
        import sqlalchemy as sa
        engine = self._get_engine()
        try:
            with engine.connect() as conn:
                yield conn
        except sa.exc.OperationalError as exc:
            log.error("db_connection_failed", error=str(exc))
            raise DatabaseConnectionError(str(exc)) from exc

    def health_check(self) -> bool:
        import sqlalchemy as sa
        try:
            with self.get_connection() as conn:
                conn.execute(sa.text("SELECT 1"))
            log.info("db_health_check_passed")
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("db_health_check_failed", error=str(exc))
            return False

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            log.info("db_engine_disposed")

    @property
    def dialect_name(self) -> str:
        mapping = {
            DatabaseType.POSTGRESQL: "PostgreSQL",
            DatabaseType.MYSQL:      "MySQL",
            DatabaseType.SNOWFLAKE:  "Snowflake",
        }
        return mapping.get(config.DATABASE_TYPE, config.DATABASE_TYPE.value)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_connection_manager() -> PrestoConnectionManager | ConnectionManager:
    """
    Return the appropriate connection manager for the configured DATABASE_TYPE.

    * ``trino``      → :class:`PrestoConnectionManager`
    * Everything else → :class:`ConnectionManager` (SQLAlchemy)
    """
    if config.DATABASE_TYPE == DatabaseType.TRINO:
        log.info("connection_manager_factory", type="PrestoConnectionManager")
        return PrestoConnectionManager()

    log.info("connection_manager_factory", type="ConnectionManager",
             db_type=config.DATABASE_TYPE.value)
    return ConnectionManager()
