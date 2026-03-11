"""
DataPilot – NLP → SQL → Result Pipeline
Orchestrates: KB load → schema fetch → prompt build → LLM → validate → execute.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from adapters.base import DatabaseAdapter, SchemaContext, QueryResult
from knowledge_base.loader import get_knowledge_base
from llm.ollama import OllamaProvider, LLMResponse
from pipeline.validator import validate_sql, ValidationResult

logger = logging.getLogger(__name__)

# Schema cache: {adapter_id → (SchemaContext, cached_at_epoch)}
_schema_cache: dict[str, tuple[SchemaContext, float]] = {}


async def get_schema_cached(
    adapter: DatabaseAdapter, ttl: int = 3600
) -> SchemaContext:
    """
    Returns cached schema if still fresh, otherwise re-fetches.
    Cache key is the adapter's source_name.
    """
    key   = adapter.source_name
    now   = time.time()
    entry = _schema_cache.get(key)
    if entry and (now - entry[1]) < ttl:
        logger.debug("Schema cache hit: %s", key)
        return entry[0]
    logger.info("Fetching schema: %s", key)
    schema = await adapter.fetch_schema()
    _schema_cache[key] = (schema, now)
    return schema


def invalidate_schema_cache(adapter: DatabaseAdapter) -> None:
    _schema_cache.pop(adapter.source_name, None)


@dataclass
class PipelineResult:
    # Input
    user_query: str
    # LLM output
    generated_sql: str
    canonical_sql: str              # After sqlglot normalisation
    model_used: str
    complexity: str
    # Validation
    validation: ValidationResult
    # Execution
    query_result: Optional[QueryResult] = None
    # Timing
    schema_fetch_ms: int  = 0
    llm_ms: int           = 0
    validation_ms: int    = 0
    execution_ms: int     = 0
    total_ms: int         = 0
    # Meta
    fallback_model_used: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return (
            self.validation.valid
            and self.query_result is not None
            and self.query_result.error is None
        )

    def to_api_response(self) -> dict:
        result_rows = []
        result_cols = []
        row_count   = 0
        truncated   = False
        exec_error  = None

        if self.query_result:
            result_rows = self.query_result.rows
            result_cols = self.query_result.columns
            row_count   = self.query_result.row_count
            truncated   = self.query_result.truncated
            exec_error  = self.query_result.error

        return {
            "success":       self.success,
            "query":         self.user_query,
            "sql":           self.canonical_sql or self.generated_sql,
            "model_used":    self.model_used,
            "complexity":    self.complexity,
            "fallback_used": self.fallback_model_used,
            "validation": {
                "valid":    self.validation.valid,
                "errors":   self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "data": {
                "columns":   result_cols,
                "rows":      result_rows,
                "row_count": row_count,
                "truncated": truncated,
            },
            "timing": {
                "schema_fetch_ms": self.schema_fetch_ms,
                "llm_ms":          self.llm_ms,
                "validation_ms":   self.validation_ms,
                "execution_ms":    self.execution_ms,
                "total_ms":        self.total_ms,
            },
            "error": self.error or exec_error,
        }


class NLPToSQLPipeline:
    """
    Main orchestrator. One instance per application lifetime.
    Adapters are injected — the pipeline doesn't care which DB it's talking to.
    """

    def __init__(
        self,
        adapter: DatabaseAdapter,
        llm: OllamaProvider,
        schema_cache_ttl: int = 3600,
    ) -> None:
        self._adapter    = adapter
        self._llm        = llm
        self._cache_ttl  = schema_cache_ttl
        # Load the knowledge base for this adapter's dialect once
        self._kb = get_knowledge_base(adapter.dialect)
        logger.info(
            "Pipeline initialised: adapter=%s | kb=%s",
            adapter.source_name, self._kb.display_name,
        )

    async def run(self, user_query: str) -> PipelineResult:
        """
        Full pipeline execution:
          1. Fetch schema (cached)
          2. Build system prompt from KB + schema
          3. Generate SQL via LLM
          4. Validate SQL (3-layer)
          5. Execute SQL
          6. Return unified result
        """
        total_start = time.monotonic()

        # ── 1. Schema ─────────────────────────────────────────────────────
        t0     = time.monotonic()
        schema = await get_schema_cached(self._adapter, self._cache_ttl)
        schema_ms = int((time.monotonic() - t0) * 1000)

        # ── 2. Prompt ─────────────────────────────────────────────────────
        system_prompt = self._kb.format_prompt(schema)

        # ── 3. LLM ────────────────────────────────────────────────────────
        t0          = time.monotonic()
        llm_resp: LLMResponse = await self._llm.generate_sql(user_query, system_prompt)
        llm_ms      = int((time.monotonic() - t0) * 1000)

        # ── 4. Validate ───────────────────────────────────────────────────
        t0         = time.monotonic()
        validation = validate_sql(llm_resp.sql, self._adapter.dialect, schema)
        val_ms     = int((time.monotonic() - t0) * 1000)

        result = PipelineResult(
            user_query    = user_query,
            generated_sql = llm_resp.sql,
            canonical_sql = validation.canonical_sql,
            model_used    = llm_resp.model_used,
            complexity    = llm_resp.complexity.value,
            validation    = validation,
            schema_fetch_ms = schema_ms,
            llm_ms          = llm_ms,
            validation_ms   = val_ms,
            fallback_model_used = llm_resp.fallback_used,
        )

        if not validation.valid:
            logger.warning(
                "Validation failed for query '%s': %s",
                user_query, validation.errors,
            )
            result.total_ms = int((time.monotonic() - total_start) * 1000)
            result.error    = "; ".join(validation.errors)
            return result

        # ── 5. Execute ────────────────────────────────────────────────────
        t0           = time.monotonic()
        query_result = await self._adapter.execute(validation.canonical_sql)
        exec_ms      = int((time.monotonic() - t0) * 1000)

        result.query_result  = query_result
        result.execution_ms  = exec_ms
        result.total_ms      = int((time.monotonic() - total_start) * 1000)

        logger.info(
            "Pipeline complete | model=%s | complexity=%s | rows=%d | total=%dms",
            llm_resp.model_used,
            llm_resp.complexity.value,
            query_result.row_count,
            result.total_ms,
        )
        return result
