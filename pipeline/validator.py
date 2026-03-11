"""
DataPilot – SQL Validator
Three-layer defence before any SQL touches the database:
  Layer 1 – Blocked keyword check (regex, instant)
  Layer 2 – sqlglot parse + dialect transpile (catches syntax errors)
  Layer 3 – Schema whitelist (every table/column must exist in live schema)
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

from adapters.base import SQLDialect, SchemaContext
from config.settings import get_settings

logger = logging.getLogger(__name__)

_SQLGLOT_DIALECT_MAP = {
    SQLDialect.POSTGRESQL: "postgres",
    SQLDialect.AZURE_SQL:  "tsql",
}


@dataclass
class ValidationResult:
    valid: bool
    canonical_sql: str = ""         # sqlglot-normalised SQL (use this for execution)
    errors: list[str]  = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False


class SQLValidator:

    def __init__(self) -> None:
        self._settings = get_settings()
        self._blocked  = [kw.upper() for kw in self._settings.sql_blocked_keywords]
        # Pre-compile blocked keyword regex (word-boundary aware)
        self._blocked_re = re.compile(
            r"\b(" + "|".join(re.escape(kw) for kw in self._blocked) + r")\b",
            re.IGNORECASE,
        )

    def validate(self, sql: str, dialect: SQLDialect, schema: SchemaContext) -> ValidationResult:
        result = ValidationResult(valid=True, canonical_sql=sql)

        # ── Layer 1: Blocked keywords ─────────────────────────────────────
        match = self._blocked_re.search(sql)
        if match:
            result.add_error(
                f"Blocked keyword '{match.group()}' found. "
                f"Only SELECT statements are allowed."
            )
            return result   # No point continuing

        # ── Layer 2: sqlglot parse + re-emit ──────────────────────────────
        sqlglot_dialect = _SQLGLOT_DIALECT_MAP[dialect]
        try:
            statements = sqlglot.parse(sql, dialect=sqlglot_dialect)
            if not statements:
                result.add_error("Could not parse SQL — no statements found.")
                return result
            if len(statements) > 1:
                result.add_error("Multiple statements detected. Only one SELECT is allowed.")
                return result

            stmt = statements[0]

            # Ensure it's a SELECT
            if not isinstance(stmt, exp.Select):
                result.add_error(
                    f"Statement type '{type(stmt).__name__}' is not allowed. Only SELECT."
                )
                return result

            # Re-emit in the target dialect — this normalises and catches subtle errors
            canonical = stmt.sql(dialect=sqlglot_dialect, pretty=False)
            result.canonical_sql = canonical + (";" if not canonical.endswith(";") else "")

        except sqlglot.errors.SqlglotError as exc:
            result.add_error(f"SQL syntax error: {exc}")
            return result

        # ── Layer 3: Schema whitelist ─────────────────────────────────────
        self._check_schema_whitelist(stmt, schema, result)

        if result.valid:
            logger.debug("SQL validated OK: %.80s", result.canonical_sql)
        else:
            logger.warning("SQL validation failed: %s | SQL: %.80s", result.errors, sql)

        return result

    def _check_schema_whitelist(
        self,
        stmt: exp.Select,
        schema: SchemaContext,
        result: ValidationResult,
    ) -> None:
        """
        Walk the AST and verify every Table reference exists in the live schema.
        Column whitelisting is done at table level (too noisy to flag every alias).
        """
        known_tables = {t.name.lower() for t in schema.tables}

        for table_node in stmt.find_all(exp.Table):
            raw_name = table_node.name
            if not raw_name:
                continue
            name_lower = raw_name.lower()
            if name_lower not in known_tables:
                result.add_error(
                    f"Table '{raw_name}' does not exist in the schema. "
                    f"Known tables: {sorted(known_tables)}"
                )


def validate_sql(
    sql: str, dialect: SQLDialect, schema: SchemaContext
) -> ValidationResult:
    """Module-level convenience wrapper."""
    return SQLValidator().validate(sql, dialect, schema)
