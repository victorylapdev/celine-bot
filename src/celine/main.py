"""FastAPI entry point for the Celine backend."""

from fastapi import FastAPI

from celine import __version__
from celine.core.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Celine",
    description="Personal technology assistant backend.",
    version=__version__,
)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return a lightweight liveness response."""
    return {"status": "ok", "environment": settings.environment}
