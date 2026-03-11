"""
DataPilot – Ollama LLM Provider (Air-Gapped)
Supports:
  - qwen2.5-coder:7b  (dev / low-resource / simple queries)
  - sqlcoder:15b       (production / complex joins / high accuracy)

Auto-routing: Analyses query complexity and routes to the appropriate model.
Fallback:     If the primary model is unavailable, falls back to the secondary.
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

from config.settings import get_settings, OllamaModel

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    SIMPLE  = "simple"   # Single table, no joins, basic aggregation
    MEDIUM  = "medium"   # 2-table join or window functions
    COMPLEX = "complex"  # Multi-join, CTE, subquery, recursive


# Signals used to estimate complexity before generation
_COMPLEX_SIGNALS = [
    r"\bjoin\b", r"\bcte\b", r"\bwith\b", r"\brecursive\b",
    r"\bwindow\b", r"\bpartition\b", r"\blag\b", r"\blead\b",
    r"\brank\b", r"\brow_number\b", r"\bsubquery\b",
    r"\bgroup by.+having\b",
]
_MEDIUM_SIGNALS = [
    r"\bjoin\b", r"\bgroup by\b", r"\bsum\b", r"\bavg\b",
    r"\bcount\b", r"\bmin\b", r"\bmax\b",
]


def estimate_complexity(user_query: str) -> QueryComplexity:
    """
    Quick heuristic — never calls the LLM, just scans the NL query.
    Used to decide which model to route to before generation starts.
    """
    q = user_query.lower()
    if any(re.search(sig, q) for sig in _COMPLEX_SIGNALS):
        return QueryComplexity.COMPLEX
    if any(re.search(sig, q) for sig in _MEDIUM_SIGNALS):
        return QueryComplexity.MEDIUM
    return QueryComplexity.SIMPLE


@dataclass
class LLMResponse:
    sql: str
    model_used: str
    complexity: QueryComplexity
    generation_ms: int
    prompt_tokens: int   = 0
    completion_tokens: int = 0
    fallback_used: bool  = False


class OllamaProvider:
    """
    Async Ollama client. Uses httpx for non-blocking HTTP.
    All Ollama endpoints are local — no external network calls.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(self._settings.llm_timeout_seconds),
        )
        # Warm up both models so first real request isn't cold
        await self._warm_up()

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _warm_up(self) -> None:
        """Pre-load models into Ollama memory at startup."""
        for model in [
            self._settings.llm_model_primary,
            self._settings.llm_model_fallback,
        ]:
            try:
                await self._ping_model(model)
                logger.info("Ollama model warmed up: %s", model)
            except Exception as exc:
                logger.warning("Could not warm up model %s: %s", model, exc)

    async def _ping_model(self, model: str) -> None:
        """Send a tiny request to load model weights into RAM."""
        await self._generate(model, "SELECT 1;", system="You are a SQL expert.")

    async def health_check(self) -> dict[str, bool]:
        """Check which models are available in Ollama."""
        results: dict[str, bool] = {}
        for model in [
            self._settings.llm_model_primary,
            self._settings.llm_model_fallback,
        ]:
            try:
                resp = await self._client.get("/api/tags")
                tags = resp.json().get("models", [])
                available = any(m["name"].startswith(model) for m in tags)
                results[model] = available
            except Exception:
                results[model] = False
        return results

    def _select_model(self, complexity: QueryComplexity) -> str:
        """
        Routing logic:
          - COMPLEX or MEDIUM → sqlcoder:15b (primary) for best accuracy
          - SIMPLE            → qwen2.5-coder:7b (fallback) is sufficient and faster
          - If auto_route is disabled → always use primary
        """
        s = self._settings
        if not s.llm_auto_route:
            return s.llm_model_primary
        if complexity == QueryComplexity.SIMPLE:
            return s.llm_model_fallback     # 7b is fine for simple queries
        return s.llm_model_primary          # 15b for anything with joins

    async def generate_sql(
        self,
        user_query: str,
        system_prompt: str,
    ) -> LLMResponse:
        """
        Main entry point. Generates SQL from NL query using the routed model.
        Falls back to the other model if the primary fails.
        """
        complexity   = estimate_complexity(user_query)
        chosen_model = self._select_model(complexity)
        fallback     = False
        start        = time.monotonic()

        logger.info(
            "Routing query [complexity=%s] → model=%s | query=%.60s",
            complexity.value, chosen_model, user_query,
        )

        try:
            raw_sql = await self._generate(chosen_model, user_query, system_prompt)
        except Exception as exc:
            logger.warning(
                "Model %s failed (%s), falling back to %s",
                chosen_model, exc,
                self._settings.llm_model_fallback
                if chosen_model == self._settings.llm_model_primary
                else self._settings.llm_model_primary,
            )
            fallback_model = (
                self._settings.llm_model_fallback
                if chosen_model == self._settings.llm_model_primary
                else self._settings.llm_model_primary
            )
            raw_sql      = await self._generate(fallback_model, user_query, system_prompt)
            chosen_model = fallback_model
            fallback     = True

        elapsed_ms = int((time.monotonic() - start) * 1000)
        cleaned    = _clean_sql_output(raw_sql)

        logger.info(
            "SQL generated in %dms using %s: %.80s",
            elapsed_ms, chosen_model, cleaned,
        )

        return LLMResponse(
            sql=cleaned,
            model_used=chosen_model,
            complexity=complexity,
            generation_ms=elapsed_ms,
            fallback_used=fallback,
        )

    async def _generate(
        self, model: str, prompt: str, system: str
    ) -> str:
        """
        Low-level Ollama /api/generate call.
        Uses stream=False so we wait for the complete response.
        """
        payload = {
            "model":  model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature":  self._settings.llm_temperature,
                "num_predict":  self._settings.llm_max_tokens,
            },
        }
        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


def _clean_sql_output(raw: str) -> str:
    """
    Extract SQL from LLM response, stripping conversational text and markdown.
    """
    # 1. Try to extract from a markdown code block first
    code_block_match = re.search(r"```(?:sql)?(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if code_block_match:
        raw = code_block_match.group(1)
        
    # 2. Remove ANSI escape sequences if the LLM hallucinated them
    raw = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", raw)
    # Also remove literal brackets with ANSI codes like [4m
    raw = re.sub(r"\[\d+m", "", raw)
    raw = re.sub(r"\[0m", "", raw)

    # 3. Strip leading/trailing whitespace
    raw = raw.strip()

    # 4. Remove any preamble before the first SELECT or WITH
    match = re.search(r"\b(SELECT|WITH)\b", raw, flags=re.IGNORECASE)
    if match:
        raw = raw[match.start():]

    # 5. Stop parsing at the first semicolon to ignore trailing conversational text
    if ";" in raw:
        raw = raw.split(";")[0]

    raw = raw.strip()
    
    # 6. Ensure single trailing semicolon
    if raw:
        raw += ";"

    return raw
