CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS celine_sessions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS celine_memories (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    session_id UUID REFERENCES celine_sessions(id) ON DELETE SET NULL,
    embedding VECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS celine_memories_category_idx ON celine_memories (category);

CREATE TABLE IF NOT EXISTS celine_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES celine_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS celine_messages_session_created_idx
    ON celine_messages (session_id, created_at);
