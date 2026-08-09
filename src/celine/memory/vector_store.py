"""Vector stores for development and PostgreSQL pgvector deployments."""

import math
import re
from collections.abc import Callable, Iterable, Sequence
from uuid import UUID

from celine.memory.models import MemoryCategory


class HashingEmbeddingGenerator:
    """Deterministic development embedding; replace with a semantic model in production."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            vector[hash(token) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: dict[UUID, tuple[list[float], MemoryCategory]] = {}

    def upsert(self, memory_id: UUID, embedding: Sequence[float], category: MemoryCategory) -> None:
        self._vectors[memory_id] = (list(embedding), category)

    def delete(self, memory_id: UUID) -> None:
        self._vectors.pop(memory_id, None)

    def search(
        self, embedding: Sequence[float], limit: int, categories: Iterable[MemoryCategory] | None = None
    ) -> list[tuple[UUID, float]]:
        allowed = set(categories) if categories else None
        results = [
            (memory_id, self._cosine(embedding, candidate))
            for memory_id, (candidate, category) in self._vectors.items()
            if allowed is None or category in allowed
        ]
        return sorted(results, key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0


class PgVectorStore:
    """pgvector adapter; expects the schema in `001_memory.sql` to be installed."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        self.connection_factory = connection_factory

    def upsert(self, memory_id: UUID, embedding: Sequence[float], category: MemoryCategory) -> None:
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        with self.connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE celine_memories SET embedding=%s::vector, category=%s WHERE id=%s",
                (vector, category.value, memory_id),
            )

    def delete(self, memory_id: UUID) -> None:
        del memory_id

    def search(
        self, embedding: Sequence[float], limit: int, categories: Iterable[MemoryCategory] | None = None
    ) -> list[tuple[UUID, float]]:
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        category_values = [category.value for category in categories] if categories else None
        query = """
            SELECT id, 1 - (embedding <=> %s::vector) AS score
            FROM celine_memories
            WHERE embedding IS NOT NULL
              AND (%s::text[] IS NULL OR category = ANY(%s::text[]))
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self.connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(query, (vector, category_values, category_values, vector, limit))
            return [(row[0], float(row[1])) for row in cursor.fetchall()]
