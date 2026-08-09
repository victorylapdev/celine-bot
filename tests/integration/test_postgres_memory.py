import os
from pathlib import Path

import psycopg
import pytest

from celine.memory.factory import build_postgres_memory_manager
from celine.memory.models import MemoryCategory

POSTGRES_URL = os.getenv("PYTEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="PYTEST_POSTGRES_URL não configurada")


@pytest.mark.integration
def test_postgres_memory_crud_and_vector_search() -> None:
    migration = Path("infra/database/migrations/001_memory.sql").read_text(encoding="utf-8")
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(migration)

    manager = build_postgres_memory_manager(POSTGRES_URL)
    memory = manager.remember("O projeto residencial usa ESP32.", MemoryCategory.PROJECT)

    assert manager.recall("ESP32")[0].memory.id == memory.id
    assert manager.update_memory(memory.id, "O projeto residencial usa ESP32-C6.").content.endswith("ESP32-C6.")
    assert manager.forget(memory.id)
