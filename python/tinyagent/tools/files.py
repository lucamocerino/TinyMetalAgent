"""File-system tools: read_file, list_dir, glob, grep, write_file, apply_patch."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, Dict, List

from . import ToolContext, ToolSpec


class ToolError(Exception):
    """Raised by a tool handler when the call cannot be completed."""


def resolve_path(ctx: ToolContext, raw: str, must_exist: bool = False) -> Path:
    """Resolve ``raw`` against the workspace root and forbid escaping it."""
    root = ctx.root.resolve()
    candidate = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ToolError(f"path escapes workspace root: {raw}")
    if must_exist and not candidate.exists():
        raise ToolError(f"no such path: {raw}")
    return candidate


def _rel(ctx: ToolContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ctx.root.resolve()))
    except ValueError:  # pragma: no cover - defensive
        return str(path)


def read_file(args: Dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_path(ctx, str(args["path"]), must_exist=True)
    if not path.is_file():
        raise ToolError(f"not a file: {args['path']}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = args.get("start_line")
    end = args.get("end_line")
    if start is not None or end is not None:
        s = int(start) - 1 if start is not None else 0
        e = int(end) if end is not None else len(lines)
        s = max(s, 0)
        lines = lines[s:e]
        offset = s + 1
    else:
        offset = 1
    numbered = [f"{offset + i}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered) if numbered else "(empty file)"


def list_dir(args: Dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_path(ctx, str(args.get("path", ".")), must_exist=True)
    if not path.is_dir():
        raise ToolError(f"not a directory: {args.get('path', '.')}")
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    out = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
    return "\n".join(out) if out else "(empty directory)"


def glob_tool(args: Dict[str, Any], ctx: ToolContext) -> str:
    pattern = str(args["pattern"])
    root = ctx.root.resolve()
    matches = [
        _rel(ctx, p)
        for p in sorted(root.glob(pattern))
    ]
    return "\n".join(matches) if matches else "(no matches)"


def grep_tool(args: Dict[str, Any], ctx: ToolContext) -> str:
    pattern = re.compile(str(args["pattern"]))
    file_glob = str(args.get("glob", "**/*"))
    base = resolve_path(ctx, str(args.get("path", ".")), must_exist=True)
    limit = int(args.get("max_results", 100))
    results: List[str] = []
    files = [base] if base.is_file() else sorted(base.glob(file_glob))
    for fp in files:
        if not fp.is_file():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                results.append(f"{_rel(ctx, fp)}:{lineno}:{line}")
                if len(results) >= limit:
                    return "\n".join(results) + "\n... [result limit reached]"
    return "\n".join(results) if results else "(no matches)"


def write_file(args: Dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_path(ctx, str(args["path"]))
    content = str(args.get("content", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {_rel(ctx, path)}"


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply unified-diff hunks (single file) to ``original`` text.

    Supports standard ``@@ -a,b +c,d @@`` hunks with space/-/+ line prefixes.
    Raises ``ToolError`` if a context/removed line does not match.
    """
    src_lines = original.splitlines()
    out: List[str] = []
    src_idx = 0
    diff_lines = diff.splitlines()
    i = 0
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    saw_hunk = False
    while i < len(diff_lines):
        line = diff_lines[i]
        if line.startswith("--- ") or line.startswith("+++ "):
            i += 1
            continue
        m = hunk_re.match(line)
        if not m:
            i += 1
            continue
        saw_hunk = True
        old_start = int(m.group(1))
        # Copy unchanged lines before the hunk.
        while src_idx < old_start - 1:
            out.append(src_lines[src_idx])
            src_idx += 1
        i += 1
        while i < len(diff_lines) and not hunk_re.match(diff_lines[i]):
            hl = diff_lines[i]
            if hl.startswith("--- ") or hl.startswith("+++ "):
                break
            if hl.startswith(" "):
                if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                    raise ToolError("patch context mismatch")
                out.append(src_lines[src_idx])
                src_idx += 1
            elif hl.startswith("-"):
                if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                    raise ToolError("patch removal mismatch")
                src_idx += 1
            elif hl.startswith("+"):
                out.append(hl[1:])
            elif hl == "":
                # treat blank diff line as a blank context line
                if src_idx < len(src_lines) and src_lines[src_idx] == "":
                    out.append("")
                    src_idx += 1
            i += 1
    if not saw_hunk:
        raise ToolError("no valid hunks in diff")
    out.extend(src_lines[src_idx:])
    trailing = "\n" if original.endswith("\n") else ""
    return "\n".join(out) + trailing


def apply_patch(args: Dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_path(ctx, str(args["path"]))
    diff = str(args["diff"])
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = apply_unified_diff(original, diff)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return f"patched {_rel(ctx, path)} ({len(updated)} bytes)"


SPECS: List[ToolSpec] = [
    ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file. Optionally restrict to a line range.",
        effect="read",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "1-based first line (optional)."},
                "end_line": {"type": "integer", "description": "1-based last line, inclusive (optional)."},
            },
            "required": ["path"],
        },
        handler=read_file,
    ),
    ToolSpec(
        name="list_dir",
        description="List the entries of a directory (directories end with '/').",
        effect="read",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path (default '.')."}},
            "required": [],
        },
        handler=list_dir,
    ),
    ToolSpec(
        name="glob",
        description="Find files matching a glob pattern (e.g. '**/*.py').",
        effect="read",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
        handler=glob_tool,
    ),
    ToolSpec(
        name="grep",
        description="Search file contents with a regex. Returns 'path:line:text'.",
        effect="read",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Directory or file (default '.')."},
                "glob": {"type": "string", "description": "Glob filter within path (default '**/*')."},
            },
            "required": ["pattern"],
        },
        handler=grep_tool,
    ),
    ToolSpec(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        effect="write",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
    ),
    ToolSpec(
        name="apply_patch",
        description="Apply a unified diff to a single file.",
        effect="write",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "diff": {"type": "string", "description": "Unified diff hunks for the file."},
            },
            "required": ["path", "diff"],
        },
        handler=apply_patch,
    ),
]
