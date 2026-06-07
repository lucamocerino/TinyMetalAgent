"""Parse tool calls emitted by the local Qwen model.

The model is small and does not always wrap calls in ``<tool_call>`` tags, so the
parser accepts three shapes, in priority order:

1. One or more ``<tool_call>{json}</tool_call>`` blocks (the canonical Qwen format).
2. A bare top-level JSON object containing ``name`` and ``arguments``.
3. A simple line-based fallback ``TOOL: name {json-args}`` (used when JSON tool
   calls prove unreliable on a given model).

Parsing never raises on malformed model output; it returns whatever it could
recover plus the assistant's free-text content with tool-call spans removed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_LINE_RE = re.compile(r"^\s*TOOL:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})?\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_TRIPLE_QUOTED_RE = re.compile(r'"""(.*?)"""', re.DOTALL)


def _repair_triple_quoted(text: str) -> str:
    """Re-encode Python-style triple-quoted string values as valid JSON strings.

    The small model frequently writes a multi-line ``content`` value as a Python
    ``\"\"\"...\"\"\"`` block, which is invalid JSON and also breaks brace matching
    (unescaped newlines and braces inside the block). Replacing each triple-quoted
    span with a properly escaped JSON string recovers the call. A no-op when the
    output contains no triple quotes.
    """
    if '"""' not in text:
        return text
    return _TRIPLE_QUOTED_RE.sub(lambda m: json.dumps(m.group(1)), text)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ParseResult:
    content: str
    tool_calls: List[ToolCall]
    malformed: List[str] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        return bool(self.tool_calls)


def _loads_lenient(blob: str) -> Any:
    blob = blob.strip()
    blob = _CODE_FENCE_RE.sub("", blob).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # One repair attempt: drop a single trailing comma before } or ].
        repaired = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(repaired)


def _extract_json_objects(text: str) -> List[str]:
    """Return the source spans of top-level ``{...}`` objects in ``text``.

    Brace-matching that ignores braces inside strings, so multiple bare JSON
    objects emitted back-to-back (as weak models sometimes do instead of using
    ``<tool_call>`` tags) are each recovered.
    """
    spans: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : idx + 1])
                    start = -1
    return spans


def _coerce_call(obj: Any, raw: str) -> ToolCall:
    if not isinstance(obj, dict) or "name" not in obj:
        raise ValueError("not a tool call object")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = _loads_lenient(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"value": args}
    return ToolCall(name=str(obj["name"]), arguments=args, raw=raw)


def parse_tool_calls(text: str) -> ParseResult:
    """Recover tool calls from raw model output. Never raises."""
    calls: List[ToolCall] = []
    malformed: List[str] = []

    text = _repair_triple_quoted(text)

    # 1. Canonical <tool_call> blocks.
    blocks = list(_TOOL_CALL_RE.finditer(text))
    if blocks:
        content = _TOOL_CALL_RE.sub("", text).strip()
        for match in blocks:
            inner = match.group(1)
            try:
                calls.append(_coerce_call(_loads_lenient(inner), match.group(0)))
            except (json.JSONDecodeError, ValueError):
                malformed.append(inner)
        return ParseResult(content=content, tool_calls=calls, malformed=malformed)

    # 2. Line-based fallback: TOOL: name {args}
    line_matches = list(_LINE_RE.finditer(text))
    if line_matches:
        content = _LINE_RE.sub("", text).strip()
        for match in line_matches:
            name = match.group(1)
            args_blob = match.group(2) or "{}"
            try:
                args = _loads_lenient(args_blob)
                if not isinstance(args, dict):
                    args = {"value": args}
                calls.append(ToolCall(name=name, arguments=args, raw=match.group(0)))
            except json.JSONDecodeError:
                malformed.append(match.group(0))
        if calls:
            return ParseResult(content=content, tool_calls=calls, malformed=malformed)

    # 3. One or more bare JSON objects with name/arguments (no wrapper tags).
    stripped = _CODE_FENCE_RE.sub("", text.strip()).strip()
    spans = _extract_json_objects(stripped)
    if spans:
        recovered: List[ToolCall] = []
        for span in spans:
            try:
                obj = _loads_lenient(span)
                recovered.append(_coerce_call(obj, span))
            except (json.JSONDecodeError, ValueError):
                continue
        if recovered:
            # Keep any free text outside the recovered JSON spans as content.
            content = stripped
            for span in spans:
                content = content.replace(span, "", 1)
            return ParseResult(content=content.strip(), tool_calls=recovered, malformed=malformed)
        if stripped.startswith("{"):
            malformed.append(stripped)

    return ParseResult(content=text.strip(), tool_calls=[], malformed=malformed)
