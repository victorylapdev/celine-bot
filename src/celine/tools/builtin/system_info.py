"""A safe diagnostic tool that reports basic host information."""

import platform
import sys
from typing import Any

from celine.tools.base import Tool, ToolInput


class SystemInfoInput(ToolInput):
    """system_info does not need input parameters."""


class SystemInfoTool(Tool[SystemInfoInput]):
    """Return non-sensitive operating-system and Python information."""

    name = "system_info"
    description = "Retorna informações básicas sobre o sistema operacional atual."
    input_model = SystemInfoInput

    def execute(self, parameters: SystemInfoInput) -> dict[str, Any]:
        return {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
        }
