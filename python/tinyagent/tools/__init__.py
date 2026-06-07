"""Tool registry, specifications, and execution context.

Each tool declares a JSON-schema parameter spec and a side-effect class
(read/write/exec/network) that drives the approval policy in ``safety``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from ..safety import Limits

Handler = Callable[[Dict[str, Any], "ToolContext"], str]


@dataclass
class ToolContext:
    """State handed to every tool handler."""

    root: Path
    limits: Limits


@dataclass
class ToolSpec:
    name: str
    description: str
    effect: str  # read | write | exec | network
    parameters: Dict[str, Any]
    handler: Handler

    def to_schema(self, compact: bool = False) -> Dict[str, Any]:
        """OpenAI-style function schema, as embedded in the Qwen <tools> block."""
        if compact:
            params = self.parameters
            required = set(params.get("required", []))
            args = [
                name if name in required else f"{name}?"
                for name in params.get("properties", {})
            ]
            return {
                "name": self.name,
                "args": args,
            }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self, compact: bool = False, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        specs = self._tools.values() if names is None else [
            self._tools[name] for name in names if name in self._tools
        ]
        return [spec.to_schema(compact=compact) for spec in specs]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


FINISH_TOOL = ToolSpec(
    name="finish",
    description="Provide the final answer to the user and end the task. "
    "Call this when the task is complete.",
    effect="read",
    parameters={
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The final answer."}},
        "required": ["answer"],
    },
    handler=lambda args, ctx: str(args.get("answer", "")),
)


def default_registry(include_network: bool = True) -> ToolRegistry:
    """Build the standard coding-agent tool catalog."""
    from . import files, shell, web

    registry = ToolRegistry()
    for spec in files.SPECS:
        registry.register(spec)
    for spec in shell.SPECS:
        registry.register(spec)
    if include_network:
        for spec in web.SPECS:
            registry.register(spec)
    registry.register(FINISH_TOOL)
    return registry
