"""High-level memory operations, independent of the Agent and database engine."""

from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from celine.memory.base import EmbeddingGenerator, MemoryRepository, VectorStore
from celine.memory.models import (
    ConversationSession,
    HistoryMessage,
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
)


class MemoryManager:
    def __init__(self, repository: MemoryRepository, vector_store: VectorStore, embeddings: EmbeddingGenerator) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embeddings = embeddings

    def remember(self, content: str, category: MemoryCategory, metadata: dict[str, Any] | None = None, session_id: UUID | None = None) -> MemoryRecord:
        memory = MemoryRecord(content=content, category=category, metadata=metadata or {}, session_id=session_id)
        stored = self.repository.create_memory(memory)
        self.vector_store.upsert(stored.id, self.embeddings.embed(content), stored.category)
        return stored

    def update_memory(self, memory_id: UUID, content: str, category: MemoryCategory | None = None, metadata: dict[str, Any] | None = None) -> MemoryRecord:
        memory = self.repository.get_memory(memory_id)
        if memory is None:
            raise KeyError(f"Memória '{memory_id}' não encontrada.")
        memory.content = content
        if category is not None:
            memory.category = category
        if metadata is not None:
            memory.metadata = metadata
        updated = self.repository.update_memory(memory)
        self.vector_store.upsert(updated.id, self.embeddings.embed(updated.content), updated.category)
        return updated

    def forget(self, memory_id: UUID) -> bool:
        deleted = self.repository.delete_memory(memory_id)
        if deleted:
            self.vector_store.delete(memory_id)
        return deleted

    def recall(self, query: str, limit: int = 5, categories: Iterable[MemoryCategory] | None = None, min_score: float = 0.0) -> list[MemorySearchResult]:
        matches = self.vector_store.search(self.embeddings.embed(query), limit=limit, categories=categories)
        results = []
        for memory_id, score in matches:
            memory = self.repository.get_memory(memory_id)
            if memory and score >= min_score:
                results.append(MemorySearchResult(memory=memory, score=score))
        return results

    def create_session(self) -> ConversationSession:
        return self.repository.create_session(ConversationSession())

    def record_history(self, session_id: UUID, messages: Sequence[HistoryMessage]) -> None:
        self.repository.create_session(ConversationSession(id=session_id))
        for message in messages:
            self.repository.append_message(message.model_copy(update={"session_id": session_id}))

    def history(self, session_id: UUID) -> list[HistoryMessage]:
        return self.repository.list_messages(session_id)

    @staticmethod
    def format_context(memories: Sequence[MemorySearchResult], max_characters: int = 2000) -> str | None:
        lines: list[str] = []
        for item in memories:
            line = f"- [{item.memory.category}] {item.memory.content}"
            if sum(len(value) + 1 for value in lines) + len(line) > max_characters:
                break
            lines.append(line)
        return "Informações relevantes registradas:\n" + "\n".join(lines) if lines else None
