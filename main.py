"""
DataPilot – FastAPI Application
Exposes /query endpoint for NLP → SQL.
The user selects their DB source and the pipeline loads the correct KB + adapter.
"""
from __future__ import annotations
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from adapters.base import SQLDialect
from adapters.postgresql import PostgreSQLAdapter
from adapters.azure_sql import AzureSQLAdapter
from config.settings import get_settings, DBSource
from knowledge_base.loader import get_all_knowledge_bases
from llm.ollama import OllamaProvider
from pipeline.nlp_to_sql import NLPToSQLPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── Application State ─────────────────────────────────────────────────────────
_adapters: dict[DBSource, PostgreSQLAdapter | AzureSQLAdapter] = {}
_pipelines: dict[DBSource, NLPToSQLPipeline] = {}
_llm: Optional[OllamaProvider] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect DBs, warm LLM. Shutdown: close all connections."""
    global _llm
    s = get_settings()

    # ── Pre-load all knowledge bases (idempotent, cached) ─────────────────
    logger.info("Pre-loading knowledge bases...")
    get_all_knowledge_bases()

    # ── Start LLM provider ────────────────────────────────────────────────
    logger.info("Starting Ollama provider...")
    _llm = OllamaProvider()
    await _llm.start()

    # ── Connect PostgreSQL ────────────────────────────────────────────────
    if s.pg_host and s.pg_user and s.pg_database:
        try:
            pg = PostgreSQLAdapter()
            await pg.connect()
            _adapters[DBSource.POSTGRESQL] = pg
            _pipelines[DBSource.POSTGRESQL] = NLPToSQLPipeline(
                adapter=pg, llm=_llm, schema_cache_ttl=s.schema_cache_ttl
            )
            logger.info("PostgreSQL adapter ready")
        except Exception as exc:
            logger.warning("PostgreSQL connection failed: %s (will be unavailable)", exc)
    else:
        logger.info("PostgreSQL not configured — skipping")

    # ── Connect Azure SQL ─────────────────────────────────────────────────
    if s.az_server and s.az_user and s.az_database:
        try:
            az = AzureSQLAdapter()
            await az.connect()
            _adapters[DBSource.AZURE_SQL] = az
            _pipelines[DBSource.AZURE_SQL] = NLPToSQLPipeline(
                adapter=az, llm=_llm, schema_cache_ttl=s.schema_cache_ttl
            )
            logger.info("Azure SQL adapter ready")
        except Exception as exc:
            logger.warning("Azure SQL connection failed: %s (will be unavailable)", exc)
    else:
        logger.info("Azure SQL not configured — skipping")

    if not _pipelines:
        logger.error("No database connections established — API will return errors")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down DataPilot...")
    for adapter in _adapters.values():
        await adapter.disconnect()
    if _llm:
        await _llm.stop()


app = FastAPI(
    title="DataPilot",
    description="NLP → SQL → Visualization — Air-gapped with Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question",
        examples=["Get me revenue for the last 30 days"],
    )
    db_source: DBSource = Field(
        ...,
        description="Which database to query",
        examples=[DBSource.POSTGRESQL, DBSource.AZURE_SQL],
    )
    schema_name: Optional[str] = Field(
        None,
        description="Override the default schema (optional)",
    )


class HealthResponse(BaseModel):
    status: str
    databases: dict[str, bool]
    models: dict[str, bool]


# ── Dependency: resolve pipeline for the requested DB ────────────────────────
def get_pipeline(db_source: DBSource) -> NLPToSQLPipeline:
    pipeline = _pipelines.get(db_source)
    if not pipeline:
        configured = [s.value for s in _pipelines]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Database '{db_source.value}' is not available. "
                f"Configured sources: {configured or 'none'}"
            ),
        )
    return pipeline


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Returns live status of all configured databases and LLM models.
    Use this to confirm the system is ready before sending queries.
    """
    db_status = {
        src.value: await adapter.health_check()
        for src, adapter in _adapters.items()
    }
    model_status = await _llm.health_check() if _llm else {}
    overall = "ok" if any(db_status.values()) else "degraded"
    return HealthResponse(status=overall, databases=db_status, models=model_status)


@app.get("/sources", tags=["System"])
async def list_sources():
    """List all configured and available database sources."""
    return {
        "sources": [
            {
                "id":           src.value,
                "name":         _adapters[src].source_name,
                "dialect":      _adapters[src].dialect.value,
                "available":    True,
            }
            for src in _pipelines
        ]
    }


@app.post("/query", tags=["Query"])
async def query(req: QueryRequest):
    """
    Main endpoint. Converts a natural language question to SQL,
    executes it against the selected database, and returns results.

    The correct SQL dialect and knowledge base are automatically loaded
    based on the db_source you select.
    """
    pipeline = get_pipeline(req.db_source)
    result   = await pipeline.run(req.query)
    response = result.to_api_response()

    if not result.success and result.validation.valid is False:
        # Validation failed — 422 so the client knows it's a bad query, not a server error
        raise HTTPException(status_code=422, detail=response)

    return response


@app.post("/schema/refresh", tags=["System"])
async def refresh_schema(db_source: DBSource):
    """Force a schema cache refresh for a specific database."""
    adapter = _adapters.get(db_source)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Source '{db_source}' not found")
    from pipeline.nlp_to_sql import invalidate_schema_cache
    invalidate_schema_cache(adapter)
    # Immediately re-fetch
    from pipeline.nlp_to_sql import get_schema_cached
    schema = await get_schema_cached(adapter, ttl=0)
    return {
        "refreshed": True,
        "tables":    len(schema.tables),
        "database":  schema.database,
    }


# ── Frontend-facing API (bridges ChatBot UI → Pipeline) ───────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: List[dict] = []
    session_id: str = "default"


def _first_available_source() -> Optional[DBSource]:
    """Return the first configured DB source, or None."""
    for src in (DBSource.AZURE_SQL, DBSource.POSTGRESQL):
        if src in _pipelines:
            return src
    return None


def _infer_chart_type(columns: list[str], rows: list[dict]) -> str:
    """Heuristic to pick the best visualisation for the frontend."""
    if not rows or not columns:
        return "none"
    if len(rows) == 1 and len(columns) == 1:
        return "number"
    if len(columns) >= 2:
        label_col = columns[0]
        value_col = columns[1]
        # Check if label column looks like dates
        first_label = str(rows[0].get(label_col, "")).lower()
        if any(kw in label_col.lower() for kw in ("date", "month", "year", "day", "time", "week")):
            return "line"
        # Check if values are numeric
        try:
            float(rows[0].get(value_col, ""))
            if len(rows) <= 10:
                return "bar"
            return "line"
        except (ValueError, TypeError):
            pass
    return "table"


def _generate_follow_ups(question: str, columns: list[str]) -> list[str]:
    """Generate contextual follow-up suggestions."""
    follow_ups = []
    q = question.lower()
    if "total" in q or "sum" in q:
        follow_ups.append("Break this down by month")
    if "top" in q:
        follow_ups.append("Show the bottom performers instead")
    if any(kw in q for kw in ("revenue", "sales", "amount")):
        follow_ups.append("What is the trend over time?")
    if len(columns) >= 2:
        follow_ups.append(f"Group by {columns[0]}")
    if not follow_ups:
        follow_ups = ["Show me more details", "Summarize this data"]
    return follow_ups[:3]


def _build_answer(question: str, columns: list[str], rows: list[dict], error: str | None) -> str:
    """Build a natural language answer from the query results."""
    if error:
        return f"I encountered an error running that query: {error}"
    if not rows:
        return "The query returned no results. Try rephrasing your question or checking the data."
    if len(rows) == 1 and len(columns) == 1:
        val = rows[0].get(columns[0], "N/A")
        return f"The result is **{val}**."
    return f"Here are the results — **{len(rows)}** row(s) across **{len(columns)}** column(s)."


@app.get("/api/health", tags=["Frontend API"])
async def api_health():
    """Health check endpoint shaped for the frontend ChatBot UI."""
    src = _first_available_source()
    if src and _llm:
        adapter = _adapters[src]
        db_ok = await adapter.health_check()
        schema = None
        try:
            from pipeline.nlp_to_sql import get_schema_cached
            schema_ctx = await get_schema_cached(adapter)
            schema = {"tables": len(schema_ctx.tables)}
        except Exception:
            schema = {"tables": 0}
        return {
            "status": "healthy" if db_ok else "degraded",
            "source": src.value,
            "schema": schema,
        }
    return {"status": "degraded", "source": None, "schema": {"tables": 0}}


@app.post("/api/ask", tags=["Frontend API"])
async def api_ask(req: AskRequest):
    """
    Main chat endpoint for the frontend.
    Takes a natural language question, runs the NLP → SQL pipeline,
    and returns a response shaped for the ChatBot UI.
    """
    src = _first_available_source()
    if not src:
        raise HTTPException(
            status_code=503,
            detail="No database is connected. Please configure your database credentials in .env",
        )

    pipeline = _pipelines[src]
    result = await pipeline.run(req.question)
    api_resp = result.to_api_response()

    columns = api_resp["data"]["columns"]
    rows = api_resp["data"]["rows"]
    error = api_resp.get("error")

    return {
        "answer": _build_answer(req.question, columns, rows, error),
        "sql": api_resp["sql"],
        "data": api_resp["data"],
        "chart_type": _infer_chart_type(columns, rows),
        "source": src.value,
        "execution_time_ms": api_resp["timing"]["total_ms"],
        "row_count": api_resp["data"]["row_count"],
        "follow_ups": _generate_follow_ups(req.question, columns),
        "cached": False,
    }


@app.get("/", response_class=FileResponse, tags=["Frontend"])
async def serve_frontend():
    """Serve the DataPilot ChatBot frontend."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("main:app", host=s.host, port=s.port, reload=s.debug)
