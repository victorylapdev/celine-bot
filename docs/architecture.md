# Architecture

## Runtime flow

1. The desktop client sends text or captured audio to the backend.
2. The speech pipeline detects the wake word and transcribes audio when needed.
3. The assistant orchestrator builds context from session state and memory.
4. An AI provider returns a response or a typed tool call.
5. The security layer validates permissions and allowlists before execution.
6. A tool invokes a browser, Windows UI, process, memory, or scheduled action.
7. Results stream to the desktop client through WebSocket and may be spoken.
8. Logs, traces, audit events, and selected memories are persisted.

## Design boundaries

- Domain orchestration must not depend directly on OpenAI, Whisper, Piper,
  Playwright, Redis, or database clients; integrations live behind adapters.
- Every tool has a Pydantic input model, explicit risk metadata, and a permission
  policy. Sensitive or destructive operations require user confirmation.
- Credentials are loaded from environment or a secret store and never committed.
- Long-running or blocking audio and automation work must not block FastAPI's
  event loop.
- WebSocket messages use versioned Pydantic schemas.
- Local and cloud AI providers expose the same internal interface.

## Suggested delivery order

1. Core settings, structured logs, FastAPI health endpoint, and WebSocket events.
2. Text conversation with one AI provider and a safe typed tool registry.
3. Speech capture, faster-whisper transcription, and TTS playback.
4. Permission-aware process, Windows, and browser automation.
5. PostgreSQL/pgvector memory, Redis session state, and APScheduler jobs.
6. Desktop UX, observability hardening, containers, and end-to-end tests.
7. ESP32/MQTT integration after the local assistant contract is stable.
