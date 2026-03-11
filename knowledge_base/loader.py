"""
DataPilot – Knowledge Base Loader
When a user selects their database, call get_knowledge_base(dialect)
to get the correct system prompt builder and schema formatter.
KBs are loaded once and cached — never re-imported per request.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol

from adapters.base import SQLDialect, SchemaContext

logger = logging.getLogger(__name__)


class DialectKB(Protocol):
    """
    Interface that every knowledge base module must satisfy.
    Both knowledge_base/postgresql/dialect.py and azure_sql/dialect.py
    implement these two functions.
    """
    def build_system_prompt(self, schema_context_str: str) -> str: ...
    def build_schema_context_string(self, schema_context: SchemaContext) -> str: ...


@dataclass(frozen=True)
class KnowledgeBase:
    dialect: SQLDialect
    display_name: str
    build_system_prompt: Callable[[str], str]
    build_schema_context_string: Callable[[SchemaContext], str]

    def format_prompt(self, schema_context: SchemaContext) -> str:
        """
        Main entry point used by the NLP pipeline.
        Converts schema → context string → full system prompt.
        """
        ctx_str = self.build_schema_context_string(schema_context)
        return self.build_system_prompt(ctx_str)


@lru_cache(maxsize=None)
def get_knowledge_base(dialect: SQLDialect) -> KnowledgeBase:
    """
    Returns the pre-loaded KnowledgeBase for the given dialect.
    lru_cache ensures modules are imported once regardless of how many
    concurrent requests come in.
    """
    if dialect == SQLDialect.POSTGRESQL:
        from knowledge_base.postgresql.dialect import (
            build_system_prompt,
            build_schema_context_string,
        )
        kb = KnowledgeBase(
            dialect=dialect,
            display_name="PostgreSQL 15",
            build_system_prompt=build_system_prompt,
            build_schema_context_string=build_schema_context_string,
        )
        logger.info("Knowledge base loaded: PostgreSQL")
        return kb

    elif dialect == SQLDialect.AZURE_SQL:
        from knowledge_base.azure_sql.dialect import (
            build_system_prompt,
            build_schema_context_string,
        )
        kb = KnowledgeBase(
            dialect=dialect,
            display_name="Azure SQL / T-SQL",
            build_system_prompt=build_system_prompt,
            build_schema_context_string=build_schema_context_string,
        )
        logger.info("Knowledge base loaded: Azure SQL")
        return kb

    raise ValueError(f"No knowledge base registered for dialect: {dialect}")


def get_all_knowledge_bases() -> dict[SQLDialect, KnowledgeBase]:
    """Pre-warm all KBs at startup so first requests aren't slow."""
    return {d: get_knowledge_base(d) for d in SQLDialect}
