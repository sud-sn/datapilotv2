"""
DataPilot – Central configuration
Loaded once at startup. All env vars are prefixed DATAPILOT_.
"""
from __future__ import annotations
from enum import Enum
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSource(str, Enum):
    POSTGRESQL = "postgresql"
    AZURE_SQL  = "azure_sql"


class OllamaModel(str, Enum):
    QWEN_7B      = "qwen2.5-coder:7b"      # Dev / low-resource
    SQLCODER_15B = "sqlcoder:15b"           # Production / best SQL accuracy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATAPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ── LLM (Ollama, air-gapped) ─────────────────────────────────────────
    ollama_base_url: str      = "http://localhost:11434"
    # Primary model for production SQL generation
    llm_model_primary: str    = OllamaModel.QWEN_7B
    # Fallback / dev model (used when primary unavailable or complexity=low)
    llm_model_fallback: str   = OllamaModel.QWEN_7B
    # Route complex queries (joins, CTEs, window fns) to primary automatically
    llm_auto_route: bool      = False
    llm_temperature: float    = 0.0     # Deterministic SQL — never change this
    llm_timeout_seconds: int  = 60      # 15b can be slow on CPU
    llm_max_tokens: int       = 1024

    # ── Database (global defaults, overridden per-connection) ────────────
    db_query_timeout: int     = 30
    max_result_rows: int      = 10_000
    include_sample_values: bool = True
    sample_value_limit: int   = 8       # How many distinct values to show LLM

    # ── PostgreSQL ───────────────────────────────────────────────────────
    pg_host: Optional[str]     = None
    pg_port: int               = 5432
    pg_user: Optional[str]     = None
    pg_password: Optional[str] = None
    pg_database: Optional[str] = None
    pg_schema: str             = "public"
    pg_ssl_mode: str           = "prefer"
    pg_pool_min: int           = 2
    pg_pool_max: int           = 10

    # ── Azure SQL ────────────────────────────────────────────────────────
    az_server: Optional[str]   = None   # e.g. myserver.database.windows.net
    az_port: int               = 1433
    az_user: Optional[str]     = None
    az_password: Optional[str] = None
    az_database: Optional[str] = None
    az_schema: str             = "dbo"
    az_driver: str             = "ODBC Driver 18 for SQL Server"
    az_encrypt: bool           = True
    az_trust_server_cert: bool = False  # True only for local dev SQL Server
    az_pool_min: int           = 2
    az_pool_max: int           = 10
    az_connection_timeout: int = 30

    # ── Caching ──────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_ttl: int      = 3600          # Schema cache TTL in seconds
    schema_cache_ttl: int = 3600        # Keep separate so it's tunable

    # ── Security ─────────────────────────────────────────────────────────
    # SQL keywords that are NEVER allowed regardless of context
    sql_blocked_keywords: list[str] = [
        "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE",
        "ALTER", "CREATE", "EXEC", "EXECUTE", "GRANT", "REVOKE",
        "MERGE", "BULK", "OPENROWSET", "OPENDATASOURCE",
        "xp_cmdshell", "sp_executesql",
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
