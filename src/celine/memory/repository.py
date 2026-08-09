"""Memory repositories. PostgreSQL is optional at runtime and injectable."""

import json
from datetime import UTC, datetime
from uuid import UUID

from celine.memory.models import ConversationSession, HistoryMessage, MemoryRecord


class InMemoryMemoryRepository:
    """Small repository for unit tests and local development."""

    def __init__(self) -> None:
        self.memories: dict[UUID, MemoryRecord] = {}
        self.sessions: dict[UUID, ConversationSession] = {}
        self.messages: dict[UUID, list[HistoryMessage]] = {}

    def create_memory(self, memory: MemoryRecord) -> MemoryRecord:
        self.memories[memory.id] = memory.model_copy(deep=True)
        return memory

    def get_memory(self, memory_id: UUID) -> MemoryRecord | None:
        memory = self.memories.get(memory_id)
        return memory.model_copy(deep=True) if memory else None

    def update_memory(self, memory: MemoryRecord) -> MemoryRecord:
        if memory.id not in self.memories:
            raise KeyError(f"Memória '{memory.id}' não encontrada.")
        memory.updated_at = datetime.now(UTC)
        self.memories[memory.id] = memory.model_copy(deep=True)
        return memory

    def delete_memory(self, memory_id: UUID) -> bool:
        return self.memories.pop(memory_id, None) is not None

    def create_session(self, session: ConversationSession) -> ConversationSession:
        self.sessions[session.id] = session
        self.messages.setdefault(session.id, [])
        return session

    def append_message(self, message: HistoryMessage) -> HistoryMessage:
        if message.session_id not in self.sessions:
            self.create_session(ConversationSession(id=message.session_id))
        self.messages[message.session_id].append(message.model_copy(deep=True))
        return message

    def list_messages(self, session_id: UUID) -> list[HistoryMessage]:
        return [message.model_copy(deep=True) for message in self.messages.get(session_id, [])]


class PostgresMemoryRepository:
    """PostgreSQL repository using a caller-provided psycopg connection factory.

    The vector column is persisted here; `PgVectorStore` performs similarity
    queries on the same table. Install `psycopg` and apply the SQL migration.
    """

    def __init__(self, connection_factory: object) -> None:
        self.connection_factory = connection_factory

    def create_memory(self, memory: MemoryRecord) -> MemoryRecord:
        query = """
            INSERT INTO celine_memories (id, content, category, metadata, session_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
        """
        self._execute(query, (memory.id, memory.content, memory.category.value, json.dumps(memory.metadata), memory.session_id, memory.created_at, memory.updated_at))
        return memory

    def get_memory(self, memory_id: UUID) -> MemoryRecord | None:
        rows = self._fetch("SELECT id, content, category, metadata, session_id, created_at, updated_at FROM celine_memories WHERE id = %s", (memory_id,))
        return self._to_memory(rows[0]) if rows else None

    def update_memory(self, memory: MemoryRecord) -> MemoryRecord:
        memory.updated_at = datetime.now(UTC)
        self._execute("UPDATE celine_memories SET content=%s, category=%s, metadata=%s::jsonb, updated_at=%s WHERE id=%s", (memory.content, memory.category.value, json.dumps(memory.metadata), memory.updated_at, memory.id))
        return memory

    def delete_memory(self, memory_id: UUID) -> bool:
        return self._execute("DELETE FROM celine_memories WHERE id = %s", (memory_id,)) > 0

    def create_session(self, session: ConversationSession) -> ConversationSession:
        self._execute("INSERT INTO celine_sessions (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING", (session.id, session.created_at))
        return session

    def append_message(self, message: HistoryMessage) -> HistoryMessage:
        self.create_session(ConversationSession(id=message.session_id))
        self._execute("INSERT INTO celine_messages (session_id, role, content, metadata, created_at) VALUES (%s, %s, %s, %s::jsonb, %s)", (message.session_id, message.role, message.content, json.dumps(message.metadata), message.created_at))
        return message

    def list_messages(self, session_id: UUID) -> list[HistoryMessage]:
        rows = self._fetch("SELECT session_id, role, content, metadata, created_at FROM celine_messages WHERE session_id=%s ORDER BY created_at", (session_id,))
        return [HistoryMessage(session_id=row[0], role=row[1], content=row[2], metadata=row[3], created_at=row[4]) for row in rows]

    def _execute(self, query: str, parameters: tuple[object, ...]) -> int:
        with self.connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return cursor.rowcount

    def _fetch(self, query: str, parameters: tuple[object, ...]) -> list[tuple[object, ...]]:
        with self.connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()

    @staticmethod
    def _to_memory(row: tuple[object, ...]) -> MemoryRecord:
        return MemoryRecord(id=row[0], content=row[1], category=row[2], metadata=row[3], session_id=row[4], created_at=row[5], updated_at=row[6])
