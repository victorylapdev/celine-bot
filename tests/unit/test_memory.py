from celine.ai.providers.deepseek import ChatMessage, ProviderResponse
from celine.core.agent import Agent
from celine.memory.manager import MemoryManager
from celine.memory.models import MemoryCategory
from celine.memory.repository import InMemoryMemoryRepository
from celine.memory.vector_store import InMemoryVectorStore
from celine.tools.registry import ToolRegistry


class SemanticTestEmbeddings:
    """Maps related phrases to the same direction for deterministic semantic tests."""

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ("esp32", "microcontrolador", "casa", "residencial")):
            return [1.0, 0.0]
        return [0.0, 1.0]


def build_manager() -> MemoryManager:
    return MemoryManager(InMemoryMemoryRepository(), InMemoryVectorStore(), SemanticTestEmbeddings())


def test_create_recall_update_and_delete_memory() -> None:
    manager = build_manager()
    memory = manager.remember("A usuária prefere respostas em português.", MemoryCategory.PREFERENCE)

    found = manager.recall("Qual idioma devo usar?")
    assert found[0].memory.id == memory.id

    updated = manager.update_memory(memory.id, "A usuária prefere respostas curtas em português.")
    assert updated.content.startswith("A usuária prefere")
    assert manager.forget(memory.id)
    assert manager.recall("Qual idioma devo usar?") == []


def test_semantic_search_and_category_filtering() -> None:
    manager = build_manager()
    project = manager.remember("O projeto de automação residencial utiliza ESP32.", MemoryCategory.PROJECT)
    manager.remember("A usuária prefere respostas em português.", MemoryCategory.PREFERENCE)

    results = manager.recall("Qual microcontrolador uso na minha casa?", categories=[MemoryCategory.PROJECT])

    assert results[0].memory.id == project.id
    assert results[0].score == 1.0


def test_history_is_stored_and_retrieved() -> None:
    manager = build_manager()
    session = manager.create_session()
    messages = [
        manager_message(session.id, "user", "Olá"),
        manager_message(session.id, "assistant", "Como posso ajudar?"),
    ]

    manager.record_history(session.id, messages)

    assert [message.content for message in manager.history(session.id)] == ["Olá", "Como posso ajudar?"]


def manager_message(session_id: object, role: str, content: str):
    from celine.memory.models import HistoryMessage

    return HistoryMessage(session_id=session_id, role=role, content=content)


class ContextCheckingProvider:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def complete(
        self, messages: list[ChatMessage], tools: tuple[dict[str, object], ...]
    ) -> ProviderResponse:
        self.messages = messages
        return ProviderResponse(message=ChatMessage(role="assistant", content="Você usa ESP32."))


def test_agent_recovers_memory_injects_context_and_persists_history() -> None:
    manager = build_manager()
    manager.remember("O projeto de automação residencial utiliza ESP32.", MemoryCategory.PROJECT)
    provider = ContextCheckingProvider()
    agent = Agent(provider, ToolRegistry(), memory_manager=manager)

    result = agent.run("Qual microcontrolador uso na minha casa?")

    assert result.succeeded
    assert provider.messages[0].role == "system"
    assert "ESP32" in (provider.messages[0].content or "")
    assert result.context.session_id is not None
    assert [message.role for message in manager.history(result.context.session_id)] == ["system", "user", "assistant"]
