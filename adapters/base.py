"""
DataPilot – Base adapter interface
Every database adapter must implement this contract.
The NLP pipeline only ever talks to this interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class SQLDialect(str, Enum):
    POSTGRESQL = "postgresql"
    AZURE_SQL  = "tsql"         # sqlglot dialect name


@dataclass
class ColumnInfo:
    name: str
    data_type: str              # Normalised type: text, numeric, timestamp, boolean, json
    raw_type: str               # Exact DB type string
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None   # "other_table.column"
    sample_values: list[Any]   = field(default_factory=list)
    description: Optional[str] = None  # From column comment / semantic layer


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[ColumnInfo]   = field(default_factory=list)
    row_count_estimate: Optional[int] = None
    description: Optional[str] = None  # From table comment / semantic layer
    # Relationships discovered via FK introspection
    foreign_keys: list[dict]    = field(default_factory=list)
    # e.g. [{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}]

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class SchemaContext:
    """
    The full schema handed to the LLM.
    Built once (or from cache) and reused across requests.
    """
    dialect: SQLDialect
    database: str
    default_schema: str
    tables: list[TableInfo]     = field(default_factory=list)
    # FK graph: {table -> [(fk_col, ref_table, ref_col)]}
    relationships: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)

    def get_table(self, name: str) -> Optional[TableInfo]:
        name_lower = name.lower()
        return next(
            (t for t in self.tables if t.name.lower() == name_lower),
            None,
        )

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_ms: int
    truncated: bool = False     # True if max_result_rows was hit
    error: Optional[str] = None


class DatabaseAdapter(ABC):
    """
    Abstract base. Subclasses implement PostgreSQL and Azure SQL variants.
    All methods are async to support non-blocking I/O.
    """

    @property
    @abstractmethod
    def dialect(self) -> SQLDialect:
        """Return the SQL dialect this adapter targets."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name shown in logs and UI (e.g. 'PostgreSQL - analytics')."""

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @abstractmethod
    async def connect(self) -> None:
        """Initialise connection pool. Called once at startup."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close all connections. Called on shutdown."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the database is reachable."""

    # ── Schema ─────────────────────────────────────────────────────────────
    @abstractmethod
    async def fetch_schema(self, schema_name: Optional[str] = None) -> SchemaContext:
        """
        Introspect the database and return a full SchemaContext.
        Implementations should:
          1. List all tables in the target schema
          2. For each table, fetch column metadata (type, nullability, PK/FK)
          3. If include_sample_values=True, fetch DISTINCT top-N values per column
          4. Resolve foreign key relationships
        """

    # ── Execution ──────────────────────────────────────────────────────────
    @abstractmethod
    async def execute(self, sql: str, timeout: Optional[int] = None) -> QueryResult:
        """
        Execute a validated read-only SQL statement and return results.
        Must enforce:
          - Read-only (SELECT only at this layer as a last defence)
          - Row cap (max_result_rows)
          - Query timeout
        """

    @abstractmethod
    async def explain(self, sql: str) -> str:
        """
        Run EXPLAIN / SHOWPLAN without executing. Used for validation.
        Returns the plan as a string (not parsed — just for error detection).
        """
