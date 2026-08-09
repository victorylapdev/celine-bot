"""Storage and embedding contracts used by the memory subsystem."""

from collections.abc import Iterable, Sequence
from typing import Protocol
from uuid import UUID

from celine.memory.models import (
    ConversationSession,
    HistoryMessage,
    MemoryCategory,
    MemoryRecord,
)


class EmbeddingGenerator(Protocol):
    def embed(self, text: str) -> list[float]: ...


class MemoryRepository(Protocol):
    def create_memory(self, memory: MemoryRecord) -> MemoryRecord: ...
    def get_memory(self, memory_id: UUID) -> MemoryRecord | None: ...
    def update_memory(self, memory: MemoryRecord) -> MemoryRecord: ...
    def delete_memory(self, memory_id: UUID) -> bool: ...
    def create_session(self, session: ConversationSession) -> ConversationSession: ...
    def append_message(self, message: HistoryMessage) -> HistoryMessage: ...
    def list_messages(self, session_id: UUID) -> list[HistoryMessage]: ...


class VectorStore(Protocol):
    def upsert(self, memory_id: UUID, embedding: Sequence[float], category: MemoryCategory) -> None: ...
    def delete(self, memory_id: UUID) -> None: ...
    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        categories: Iterable[MemoryCategory] | None = None,
    ) -> list[tuple[UUID, float]]: ...
