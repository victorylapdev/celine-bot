"""Command-line utilities for Celine."""

import argparse

from celine.ai.providers.deepseek import DeepSeekClient
from celine.assistant.orchestration import ToolCallingOrchestrator
from celine.tools.builtin import SystemInfoTool
from celine.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    """Compose the tools enabled for the local Celine runtime."""
    registry = ToolRegistry()
    registry.register(SystemInfoTool())
    return registry


def main() -> None:
    """Send one prompt to DeepSeek and print its reply."""
    parser = argparse.ArgumentParser(description="Envie uma mensagem para a DeepSeek.")
    parser.add_argument("message", help="Mensagem que será enviada à DeepSeek.")
    args = parser.parse_args()

    answer = ToolCallingOrchestrator(DeepSeekClient(), build_default_registry()).respond(args.message)
    print(answer)


if __name__ == "__main__":
    main()
