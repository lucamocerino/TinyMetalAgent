"""Qwen2.5 ChatML prompt rendering (system/user/assistant/tool roles + tools).

The local TinyEngine model is a Qwen2.5-Instruct GGUF, so prompts must follow the
exact ChatML format the model was trained on. This module is the single source of
truth for turning a structured message list (plus optional tool schemas) into the
raw string fed to ``Model.generate_raw``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

Message = Dict[str, Any]
Tool = Dict[str, Any]
_PREFIX_CACHE_MAX = 32
_PREFIX_CACHE: Dict[tuple[str, tuple[str, ...]], str] = {}

DEFAULT_SYSTEM = "You are a helpful coding assistant."

_TOOLS_PREAMBLE = (
    "\n\n# Tools\n"
    "ACT mode: return one call like "
    '<tool_call>{{"name":"write_file","arguments":{{"path":"x","content":"..."}}}}</tool_call>\n'
    "<tools>\n{tools}\n</tools>\n\n"
)


def _render_system_with_tools(system_content: str, tools: Sequence[Tool]) -> str:
    rendered_tools = tuple(json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in tools)
    key = (system_content, rendered_tools)
    cached = _PREFIX_CACHE.get(key)
    if cached is not None:
        return cached
    tool_lines = "\n".join(rendered_tools)
    rendered = system_content + _TOOLS_PREAMBLE.format(tools=tool_lines)
    if len(_PREFIX_CACHE) >= _PREFIX_CACHE_MAX:
        _PREFIX_CACHE.pop(next(iter(_PREFIX_CACHE)))
    _PREFIX_CACHE[key] = rendered
    return rendered


def _render_assistant(message: Message) -> str:
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        parts: List[str] = []
        if content:
            parts.append(content)
        for call in tool_calls:
            payload = {"name": call["name"], "arguments": call.get("arguments", {})}
            parts.append(
                "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
            )
        body = "\n".join(parts)
    else:
        body = content
    return f"<|im_start|>assistant\n{body}<|im_end|>\n"


def render_chat(
    messages: Sequence[Message],
    tools: Optional[Sequence[Tool]] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages into a Qwen2.5 ChatML string.

    ``messages`` is an ordered list of ``{"role": ..., "content": ...}`` dicts.
    Assistant messages may carry ``tool_calls`` (list of ``{"name", "arguments"}``).
    ``tool`` messages are rendered as Qwen ``<tool_response>`` blocks (inside a user
    turn, matching the official template). ``tools`` are injected into the system
    message. If ``add_generation_prompt`` is set, an open assistant turn is appended.
    """
    msgs = list(messages)
    out: List[str] = []

    if tools:
        if msgs and msgs[0].get("role") == "system":
            system_content = msgs[0].get("content") or DEFAULT_SYSTEM
            body = msgs[1:]
        else:
            system_content = DEFAULT_SYSTEM
            body = msgs
        out.append(
            f"<|im_start|>system\n{_render_system_with_tools(system_content, tools)}<|im_end|>\n"
        )
    else:
        body = msgs

    for message in body:
        role = message.get("role")
        if role == "tool":
            content = message.get("content") or ""
            out.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            )
        elif role == "assistant":
            out.append(_render_assistant(message))
        elif role in ("system", "user"):
            out.append(f"<|im_start|>{role}\n{message.get('content') or ''}<|im_end|>\n")
        else:
            raise ValueError(f"unknown message role: {role!r}")

    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)
