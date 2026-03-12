"""
DataPilot – Azure SQL Adapter
Uses pyodbc with ODBC Driver 18 for SQL Server.
Connection pooling is handled via a manual pool since pyodbc is synchronous —
we run it in a thread executor to avoid blocking the event loop.
"""
from __future__ import annotations
import asyncio
import logging
import queue
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

import pyodbc

from adapters.base import (
    DatabaseAdapter, SQLDialect, SchemaContext, TableInfo,
    ColumnInfo, QueryResult,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Map T-SQL system type names → normalised types
_TSQL_TYPE_MAP: dict[str, str] = {
    # Text
    "char": "text", "nchar": "text", "varchar": "text", "nvarchar": "text",
    "text": "text", "ntext": "text", "uniqueidentifier": "text",
    "xml": "text", "sysname": "text",
    # Numeric
    "tinyint": "numeric", "smallint": "numeric", "int": "numeric",
    "bigint": "numeric", "decimal": "numeric", "numeric": "numeric",
    "float": "numeric", "real": "numeric", "money": "numeric",
    "smallmoney": "numeric", "bit": "boolean",
    # Timestamp / date
    "date": "date", "time": "time", "datetime": "timestamp",
    "datetime2": "timestamp", "smalldatetime": "timestamp",
    "datetimeoffset": "timestamp",
    # Binary / other
    "binary": "binary", "varbinary": "binary", "image": "binary",
    "rowversion": "binary", "timestamp": "binary",  # T-SQL 'timestamp' is rowversion!
    # JSON (SQL Server 2016+)
    "json": "json",
}


def _normalise_tsql_type(raw: str) -> str:
    return _TSQL_TYPE_MAP.get(raw.lower(), raw.lower())


def _bracket(name: str) -> str:
    """Wrap an identifier in square brackets for T-SQL safety."""
    return f"[{name.replace(']', ']]')}]"


class _ConnectionPool:
    """
    Simple blocking connection pool for pyodbc.
    pyodbc is synchronous; we keep a pool and run queries in asyncio's
    thread executor so we never block the event loop.
    """

    def __init__(self, connection_string: str, min_size: int, max_size: int) -> None:
        self._conn_str   = connection_string
        self._max_size   = max_size
        self._pool: queue.Queue[pyodbc.Connection] = queue.Queue(maxsize=max_size)
        self._size       = 0
        self._lock       = asyncio.Lock()

        # Pre-create min connections synchronously
        for _ in range(min_size):
            conn = pyodbc.connect(connection_string)
            self._pool.put(conn)
            self._size += 1

    @contextmanager
    def acquire(self, timeout: int = 30) -> Generator[pyodbc.Connection, None, None]:
        try:
            conn = self._pool.get(timeout=timeout)
        except queue.Empty:
            if self._size < self._max_size:
                conn = pyodbc.connect(self._conn_str)
                self._size += 1
            else:
                raise TimeoutError("All Azure SQL connections are busy")
        try:
            yield conn
        except pyodbc.Error:
            # If the connection is broken, replace it
            try:
                conn.close()
            except Exception:
                pass
            conn = pyodbc.connect(self._conn_str)
            raise
        finally:
            self._pool.put(conn)

    def close_all(self) -> None:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass


class AzureSQLAdapter(DatabaseAdapter):

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: Optional[_ConnectionPool] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def dialect(self) -> SQLDialect:
        return SQLDialect.AZURE_SQL

    @property
    def source_name(self) -> str:
        s = self._settings
        return f"Azure SQL — {s.az_server}/{s.az_database}"

    def _build_connection_string(self) -> str:
        s = self._settings
        parts = [
            f"DRIVER={{{s.az_driver}}}",
            f"SERVER={s.az_server},{s.az_port}",
            f"DATABASE={s.az_database}",
            f"UID={s.az_user}",
            f"PWD={s.az_password}",
            f"Encrypt={'yes' if s.az_encrypt else 'no'}",
            f"TrustServerCertificate={'yes' if s.az_trust_server_cert else 'no'}",
            f"Connection Timeout={s.az_connection_timeout}",
        ]
        return ";".join(parts)

    # ── Lifecycle ──────────────────────────────────────────────────────────
    async def connect(self) -> None:
        s          = self._settings
        conn_str   = self._build_connection_string()
        self._loop = asyncio.get_event_loop()
        # Pool creation is sync — run in executor
        self._pool = await self._loop.run_in_executor(
            None,
            lambda: _ConnectionPool(conn_str, s.az_pool_min, s.az_pool_max),
        )
        logger.info("Azure SQL pool created: %s", self.source_name)

    async def disconnect(self) -> None:
        if self._pool:
            await asyncio.get_event_loop().run_in_executor(
                None, self._pool.close_all
            )
            logger.info("Azure SQL pool closed")

    async def health_check(self) -> bool:
        try:
            await self._run_sync(lambda conn: conn.execute("SELECT 1").fetchone())
            return True
        except Exception as exc:
            logger.warning("Azure SQL health check failed: %s", exc)
            return False

    # ── Helpers ────────────────────────────────────────────────────────────
    async def _run_sync(self, fn):
        """Run a synchronous pyodbc operation in the thread executor.
        fn receives a pyodbc.Connection acquired from the pool."""
        loop = asyncio.get_event_loop()
        def _with_conn():
            with self._pool.acquire() as conn:
                return fn(conn)
        return await loop.run_in_executor(None, _with_conn)

    def _exec_sync(self, conn: pyodbc.Connection, sql: str, timeout: int) -> tuple[list[str], list[dict]]:
        """Blocking SQL execution. Called inside run_in_executor."""
        conn.timeout = timeout
        cursor = conn.cursor()
        cursor.execute(sql)
        cols  = [desc[0] for desc in cursor.description] if cursor.description else []
        rows  = []
        limit = self._settings.max_result_rows
        for row in cursor:
            rows.append(dict(zip(cols, row)))
            if len(rows) >= limit:
                break
        return cols, rows

    # ── Schema ─────────────────────────────────────────────────────────────
    async def fetch_schema(self, schema_name: Optional[str] = None) -> SchemaContext:
        s      = self._settings
        schema = schema_name or s.az_schema

        tables   = await self._run_sync(lambda conn: self._fetch_tables_sync(conn, schema))
        fk_graph = await self._run_sync(lambda conn: self._fetch_fk_graph_sync(conn, schema))

        ctx = SchemaContext(
            dialect=self.dialect,
            database=s.az_database,
            default_schema=schema,
            tables=tables,
            relationships=fk_graph,
        )
        logger.info(
            "Azure SQL schema fetched: %d tables in '%s'", len(tables), schema
        )
        return ctx

    def _fetch_tables_sync(
        self, conn: pyodbc.Connection, schema: str
    ) -> list[TableInfo]:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = ?
              AND TABLE_TYPE   = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """,
            schema,
        )
        tables: list[TableInfo] = []
        for (tname,) in cursor.fetchall():
            cols    = self._fetch_columns_sync(conn, schema, tname)
            row_est = self._estimate_row_count_sync(conn, schema, tname)
            fks     = self._fetch_table_fks_sync(conn, schema, tname)
            tables.append(
                TableInfo(
                    schema=schema,
                    name=tname,
                    columns=cols,
                    row_count_estimate=row_est,
                    foreign_keys=fks,
                )
            )
        return tables

    def _fetch_columns_sync(
        self, conn: pyodbc.Connection, schema: str, table: str
    ) -> list[ColumnInfo]:
        s      = self._settings
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_PK,
                CASE WHEN fk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_FK,
                fk.REF_TABLE,
                fk.REF_COLUMN,
                ep.value AS col_description
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
                  ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                  AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
            ) pk ON pk.COLUMN_NAME = c.COLUMN_NAME
            LEFT JOIN (
                SELECT
                    ku.COLUMN_NAME,
                    ccu.TABLE_NAME  AS REF_TABLE,
                    ccu.COLUMN_NAME AS REF_COLUMN
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
                  ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                  ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
            ) fk ON fk.COLUMN_NAME = c.COLUMN_NAME
            LEFT JOIN sys.extended_properties ep
              ON ep.major_id = OBJECT_ID(? + '.' + ?)
             AND ep.minor_id = c.ORDINAL_POSITION
             AND ep.name = 'MS_Description'
             AND ep.class = 1
            WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
            ORDER BY c.ORDINAL_POSITION
            """,
            schema, table,   # PK subquery
            schema, table,   # FK subquery
            schema, table,   # extended_properties
            schema, table,   # outer WHERE
        )

        columns: list[ColumnInfo] = []
        for row in cursor.fetchall():
            col_name, data_type, nullable, is_pk, is_fk, ref_tbl, ref_col, desc = row
            sample_vals: list[Any] = []
            if s.include_sample_values:
                sample_vals = self._fetch_sample_values_sync(
                    conn, schema, table, col_name, s.sample_value_limit
                )
            columns.append(
                ColumnInfo(
                    name=col_name,
                    data_type=_normalise_tsql_type(data_type),
                    raw_type=data_type,
                    nullable=nullable == "YES",
                    is_primary_key=bool(is_pk),
                    is_foreign_key=bool(is_fk),
                    references=f"{ref_tbl}.{ref_col}" if ref_tbl else None,
                    sample_values=sample_vals,
                    description=str(desc) if desc else None,
                )
            )
        return columns

    def _fetch_sample_values_sync(
        self,
        conn: pyodbc.Connection,
        schema: str,
        table: str,
        column: str,
        limit: int,
    ) -> list[Any]:
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT DISTINCT TOP {limit} {_bracket(column)}
                FROM {_bracket(schema)}.{_bracket(table)}
                WHERE {_bracket(column)} IS NOT NULL
                """
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def _estimate_row_count_sync(
        self, conn: pyodbc.Connection, schema: str, table: str
    ) -> Optional[int]:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT SUM(p.rows)
                FROM sys.tables t
                JOIN sys.schemas s  ON s.schema_id = t.schema_id
                JOIN sys.partitions p ON p.object_id = t.object_id
                WHERE s.name = ? AND t.name = ? AND p.index_id IN (0, 1)
                """,
                schema, table,
            )
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] else None
        except Exception:
            return None

    def _fetch_table_fks_sync(
        self, conn: pyodbc.Connection, schema: str, table: str
    ) -> list[dict]:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                kcu.COLUMN_NAME,
                ccu.TABLE_NAME  AS REF_TABLE,
                ccu.COLUMN_NAME AS REF_COLUMN
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
              ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
              AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
            """,
            schema, table,
        )
        return [
            {"column": r[0], "ref_table": r[1], "ref_column": r[2]}
            for r in cursor.fetchall()
        ]

    def _fetch_fk_graph_sync(
        self, conn: pyodbc.Connection, schema: str
    ) -> dict[str, list[tuple[str, str, str]]]:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                kcu.TABLE_NAME,
                kcu.COLUMN_NAME,
                ccu.TABLE_NAME  AS REF_TABLE,
                ccu.COLUMN_NAME AS REF_COLUMN
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
              ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
              AND tc.TABLE_SCHEMA = ?
            """,
            schema,
        )
        graph: dict[str, list[tuple[str, str, str]]] = {}
        for tbl, col, ref_tbl, ref_col in cursor.fetchall():
            graph.setdefault(tbl, []).append((col, ref_tbl, ref_col))
        return graph

    # ── Execution ──────────────────────────────────────────────────────────
    async def execute(self, sql: str, timeout: Optional[int] = None) -> QueryResult:
        s       = self._settings
        timeout = timeout or s.db_query_timeout
        start   = time.monotonic()

        try:
            cols, rows = await self._run_sync(
                lambda conn: self._exec_sync(conn, sql, timeout)
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            truncated  = len(rows) >= s.max_result_rows

            return QueryResult(
                sql=sql,
                columns=cols,
                rows=rows,
                row_count=len(rows),
                execution_ms=elapsed_ms,
                truncated=truncated,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("Azure SQL execution error: %s", exc)
            return QueryResult(
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                execution_ms=elapsed_ms,
                error=str(exc),
            )

    async def explain(self, sql: str) -> str:
        """Use SET SHOWPLAN_TEXT — does not execute, just produces plan."""
        def _plan(conn: pyodbc.Connection) -> str:
            cursor = conn.cursor()
            cursor.execute("SET SHOWPLAN_TEXT ON")
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return "\n".join(str(r[0]) for r in rows)
            finally:
                cursor.execute("SET SHOWPLAN_TEXT OFF")

        try:
            return await self._run_sync(_plan)
        except pyodbc.Error as exc:
            raise ValueError(f"SHOWPLAN failed: {exc}") from exc
