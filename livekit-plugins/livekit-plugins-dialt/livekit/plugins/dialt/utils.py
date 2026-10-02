from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dialt import ToolDefinition
from livekit.agents import llm
from livekit.agents.llm.utils import function_arguments_to_pydantic_model

from .models import ToolOptions


def to_dialt_tools(
    tools: list[llm.Tool],
    tool_options: dict[str, ToolOptions] | None = None,
) -> list[ToolDefinition]:
    """Convert LiveKit function tools to Dialt tool definitions.

    Provider tools are intentionally ignored because Dialt cannot execute them.
    """

    definitions: list[ToolDefinition] = []
    options = tool_options or {}
    for tool in tools:
        definition: dict[str, Any]
        if isinstance(tool, llm.RawFunctionTool):
            raw_info = tool.info
            definition = {
                "name": raw_info.name,
                "parameters": raw_info.raw_schema.get("parameters", {}),
            }
            description = raw_info.raw_schema.get("description")
            if isinstance(description, str):
                definition["description"] = description
        elif isinstance(tool, llm.FunctionTool):
            function_info = tool.info
            definition = {
                "name": function_info.name,
                "parameters": function_arguments_to_pydantic_model(tool).model_json_schema(),
            }
            if function_info.description is not None:
                definition["description"] = function_info.description
        else:
            continue

        if configured := options.get(definition["name"]):
            definition.update(
                {key: value for key, value in asdict(configured).items() if value is not None}
            )
        definitions.append(definition)  # type: ignore[arg-type]

    return definitions
