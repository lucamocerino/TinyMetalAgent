"""Shell execution tool (exec effect)."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from . import ToolContext, ToolSpec
from .files import ToolError, resolve_path


def run_shell(args: Dict[str, Any], ctx: ToolContext) -> str:
    cmd = str(args["cmd"])
    cwd = ctx.root
    if args.get("cwd"):
        cwd = resolve_path(ctx, str(args["cwd"]), must_exist=True)
    timeout = float(args.get("timeout", ctx.limits.shell_timeout))
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout:.0f}s")
    parts: List[str] = [f"exit_code: {proc.returncode}"]
    if proc.stdout:
        parts.append("stdout:\n" + proc.stdout.rstrip("\n"))
    if proc.stderr:
        parts.append("stderr:\n" + proc.stderr.rstrip("\n"))
    return "\n".join(parts)


SPECS: List[ToolSpec] = [
    ToolSpec(
        name="run_shell",
        description="Run a shell command in the workspace and capture its output. "
        "Use for builds, tests, git, running scripts.",
        effect="exec",
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The shell command to run."},
                "cwd": {"type": "string", "description": "Working directory (default workspace root)."},
                "timeout": {"type": "number", "description": "Timeout in seconds."},
            },
            "required": ["cmd"],
        },
        handler=run_shell,
    ),
]
