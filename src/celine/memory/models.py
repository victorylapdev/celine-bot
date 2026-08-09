"""Models for durable memories and conversation history."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryCategory(StrEnum):
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    FACT = "fact"
    INSTRUCTION = "instruction"
    PROJECT = "project"
    CONTEXT = "context"


class MemoryRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    category: MemoryCategory
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemorySearchResult(BaseModel):
    memory: MemoryRecord
    score: float


class ConversationSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HistoryMessage(BaseModel):
    session_id: UUID
    role: str
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
