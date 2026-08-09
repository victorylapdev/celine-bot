"""Composition helpers for PostgreSQL-backed memory."""

from collections.abc import Callable

import psycopg

from celine.memory.manager import MemoryManager
from celine.memory.repository import PostgresMemoryRepository
from celine.memory.vector_store import HashingEmbeddingGenerator, PgVectorStore


def build_postgres_memory_manager(database_url: str) -> MemoryManager:
    """Build the production memory stack after the pgvector migration is applied."""
    connection_factory: Callable[[], psycopg.Connection] = lambda: psycopg.connect(database_url)
    return MemoryManager(
        repository=PostgresMemoryRepository(connection_factory),
        vector_store=PgVectorStore(connection_factory),
        embeddings=HashingEmbeddingGenerator(),
    )
