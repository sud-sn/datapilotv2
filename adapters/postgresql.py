"""
DataPilot – PostgreSQL Adapter
Uses asyncpg for async connection pooling.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Optional

import asyncpg

from adapters.base import (
    DatabaseAdapter, SQLDialect, SchemaContext, TableInfo,
    ColumnInfo, QueryResult,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Map PostgreSQL type OIDs / type names → normalised types
_PG_TYPE_MAP: dict[str, str] = {
    # Text
    "text": "text", "varchar": "text", "character varying": "text",
    "char": "text", "bpchar": "text", "name": "text", "uuid": "text",
    # Numeric
    "int2": "numeric", "int4": "numeric", "int8": "numeric",
    "smallint": "numeric", "integer": "numeric", "bigint": "numeric",
    "numeric": "numeric", "decimal": "numeric",
    "float4": "numeric", "float8": "numeric", "real": "numeric",
    "double precision": "numeric", "money": "numeric",
    # Timestamp / date
    "timestamp": "timestamp", "timestamptz": "timestamp",
    "timestamp without time zone": "timestamp",
    "timestamp with time zone": "timestamp",
    "date": "date", "time": "time", "timetz": "time",
    "interval": "interval",
    # Boolean
    "bool": "boolean", "boolean": "boolean",
    # JSON
    "json": "json", "jsonb": "json",
    # Arrays & misc
    "array": "array", "bytea": "binary",
}


def _normalise_pg_type(raw: str) -> str:
    return _PG_TYPE_MAP.get(raw.lower(), raw.lower())


class PostgreSQLAdapter(DatabaseAdapter):

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: Optional[asyncpg.Pool] = None

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def dialect(self) -> SQLDialect:
        return SQLDialect.POSTGRESQL

    @property
    def source_name(self) -> str:
        s = self._settings
        return f"PostgreSQL — {s.pg_host}:{s.pg_port}/{s.pg_database}"

    # ── Lifecycle ──────────────────────────────────────────────────────────
    async def connect(self) -> None:
        s = self._settings
        dsn = (
            f"postgresql://{s.pg_user}:{s.pg_password}"
            f"@{s.pg_host}:{s.pg_port}/{s.pg_database}"
        )
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=s.pg_pool_min,
            max_size=s.pg_pool_max,
            command_timeout=s.db_query_timeout,
            # Enforce read-only at the session level — belt-and-suspenders
            server_settings={"default_transaction_read_only": "on"},
        )
        logger.info("PostgreSQL pool created: %s", self.source_name)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL pool closed")

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:
            logger.warning("PostgreSQL health check failed: %s", exc)
            return False

    # ── Schema ─────────────────────────────────────────────────────────────
    async def fetch_schema(self, schema_name: Optional[str] = None) -> SchemaContext:
        s      = self._settings
        schema = schema_name or s.pg_schema

        async with self._pool.acquire() as conn:
            tables      = await self._fetch_tables(conn, schema)
            fk_graph    = await self._fetch_fk_graph(conn, schema)

        ctx = SchemaContext(
            dialect=self.dialect,
            database=s.pg_database,
            default_schema=schema,
            tables=tables,
            relationships=fk_graph,
        )
        logger.info(
            "PostgreSQL schema fetched: %d tables in '%s'", len(tables), schema
        )
        return ctx

    async def _fetch_tables(
        self, conn: asyncpg.Connection, schema: str
    ) -> list[TableInfo]:
        # Fetch table names
        table_rows = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = $1
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            schema,
        )
        tables: list[TableInfo] = []
        for tr in table_rows:
            tname   = tr["table_name"]
            cols    = await self._fetch_columns(conn, schema, tname)
            row_est = await self._estimate_row_count(conn, schema, tname)
            tables.append(
                TableInfo(
                    schema=schema,
                    name=tname,
                    columns=cols,
                    row_count_estimate=row_est,
                )
            )
        return tables

    async def _fetch_columns(
        self, conn: asyncpg.Connection, schema: str, table: str
    ) -> list[ColumnInfo]:
        s = self._settings

        rows = await conn.fetch(
            """
            SELECT
                c.column_name,
                c.data_type,
                c.udt_name,
                c.is_nullable,
                c.column_default,
                col_description(
                    (quote_ident($1) || '.' || quote_ident($2))::regclass,
                    c.ordinal_position
                ) AS col_comment
            FROM information_schema.columns c
            WHERE c.table_schema = $1
              AND c.table_name   = $2
            ORDER BY c.ordinal_position
            """,
            schema, table,
        )

        # Get PK columns
        pk_rows = await conn.fetch(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = $1
              AND tc.table_name      = $2
            """,
            schema, table,
        )
        pk_cols = {r["column_name"] for r in pk_rows}

        # Get FK columns
        fk_rows = await conn.fetch(
            """
            SELECT
                kcu.column_name,
                ccu.table_name  AS ref_table,
                ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema    = $1
              AND tc.table_name      = $2
            """,
            schema, table,
        )
        fk_map = {
            r["column_name"]: f"{r['ref_table']}.{r['ref_column']}"
            for r in fk_rows
        }

        columns: list[ColumnInfo] = []
        for row in rows:
            col_name = row["column_name"]
            raw_type = row["udt_name"] or row["data_type"]

            sample_vals: list[Any] = []
            if s.include_sample_values:
                sample_vals = await self._fetch_sample_values(
                    conn, schema, table, col_name, s.sample_value_limit
                )

            columns.append(
                ColumnInfo(
                    name=col_name,
                    data_type=_normalise_pg_type(raw_type),
                    raw_type=raw_type,
                    nullable=row["is_nullable"] == "YES",
                    is_primary_key=col_name in pk_cols,
                    is_foreign_key=col_name in fk_map,
                    references=fk_map.get(col_name),
                    sample_values=sample_vals,
                    description=row["col_comment"],
                )
            )
        return columns

    async def _fetch_sample_values(
        self,
        conn: asyncpg.Connection,
        schema: str,
        table: str,
        column: str,
        limit: int,
    ) -> list[Any]:
        try:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT {conn.quote_ident(column)} AS v
                FROM {conn.quote_ident(schema)}.{conn.quote_ident(table)}
                WHERE {conn.quote_ident(column)} IS NOT NULL
                LIMIT {limit}
                """
            )
            return [r["v"] for r in rows]
        except Exception:
            return []

    async def _estimate_row_count(
        self, conn: asyncpg.Connection, schema: str, table: str
    ) -> Optional[int]:
        try:
            row = await conn.fetchrow(
                """
                SELECT reltuples::bigint AS estimate
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1 AND c.relname = $2
                """,
                schema, table,
            )
            return int(row["estimate"]) if row else None
        except Exception:
            return None

    async def _fetch_fk_graph(
        self, conn: asyncpg.Connection, schema: str
    ) -> dict[str, list[tuple[str, str, str]]]:
        rows = await conn.fetch(
            """
            SELECT
                kcu.table_name,
                kcu.column_name,
                ccu.table_name  AS ref_table,
                ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema    = $1
            """,
            schema,
        )
        graph: dict[str, list[tuple[str, str, str]]] = {}
        for r in rows:
            graph.setdefault(r["table_name"], []).append(
                (r["column_name"], r["ref_table"], r["ref_column"])
            )
        return graph

    # ── Execution ──────────────────────────────────────────────────────────
    async def execute(self, sql: str, timeout: Optional[int] = None) -> QueryResult:
        s       = self._settings
        timeout = timeout or s.db_query_timeout
        start   = time.monotonic()

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, timeout=timeout)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            truncated  = len(rows) >= s.max_result_rows

            if truncated:
                logger.warning(
                    "Result truncated to %d rows for query: %.80s",
                    s.max_result_rows, sql,
                )

            col_names = list(rows[0].keys()) if rows else []
            result_rows = [dict(r) for r in rows[: s.max_result_rows]]

            return QueryResult(
                sql=sql,
                columns=col_names,
                rows=result_rows,
                row_count=len(result_rows),
                execution_ms=elapsed_ms,
                truncated=truncated,
            )

        except asyncpg.PostgresError as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("PostgreSQL execution error: %s", exc)
            return QueryResult(
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                execution_ms=elapsed_ms,
                error=str(exc),
            )

    async def explain(self, sql: str) -> str:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(f"EXPLAIN {sql}")
            return "\n".join(r[0] for r in rows)
        except asyncpg.PostgresError as exc:
            raise ValueError(f"EXPLAIN failed: {exc}") from exc
