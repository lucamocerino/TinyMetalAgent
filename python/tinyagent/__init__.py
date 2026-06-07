"""tinyagent: a local-model terminal coding agent powered by TinyEngine + Qwen.

Inference is 100% local (a Qwen GGUF via TinyEngine); the agent's *tools* may use
the filesystem, shell, and network, like other terminal coding agents.
"""
from __future__ import annotations

from .agent import Agent, AgentEvent, AgentResult, DEFAULT_SYSTEM_PROMPT
from .engine import Engine, TinyEngine
from .safety import Approver, AuditLog, Limits
from .session import Session
from .template import render_chat
from .toolcall import ToolCall, parse_tool_calls
from .tools import ToolContext, ToolRegistry, ToolSpec, default_registry

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentResult",
    "DEFAULT_SYSTEM_PROMPT",
    "Engine",
    "TinyEngine",
    "Approver",
    "AuditLog",
    "Limits",
    "Session",
    "render_chat",
    "ToolCall",
    "parse_tool_calls",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "default_registry",
    "__version__",
]
