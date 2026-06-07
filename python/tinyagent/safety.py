"""Approval gating, execution guards, and audit logging for agent tools.

Tools are classified by side effect. Read and network effects are auto-allowed by
default (the agent is meant to browse the system and the internet freely, like
other terminal coding agents); write and exec effects require approval. Approval can
be: interactive (``ask``), auto (``yes``), or denied/simulated (``dry_run``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

Effect = str  # one of: read, write, exec, network

AUTO_ALLOWED_DEFAULT: Set[Effect] = {"read", "network"}
GATED_EFFECTS: Set[Effect] = {"write", "exec"}

# Interactive prompt returns one of these.
APPROVE_ONCE = "y"
DENY = "n"
APPROVE_SESSION = "a"


@dataclass
class Limits:
    shell_timeout: float = 30.0
    max_output_chars: int = 8000
    max_steps: int = 12


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    simulated: bool = False  # dry-run: approved-in-spirit but not executed


@dataclass
class AuditEntry:
    step: int
    name: str
    effect: Effect
    arguments: Dict[str, object]
    allowed: bool
    simulated: bool
    result_summary: str = ""


class Approver:
    """Decides whether a tool call may run, given its side-effect class."""

    def __init__(
        self,
        mode: str = "ask",
        no_network: bool = False,
        auto_allowed: Optional[Set[Effect]] = None,
        prompt_fn: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        if mode not in ("ask", "yes", "dry_run"):
            raise ValueError(f"invalid approval mode: {mode}")
        self.mode = mode
        self.no_network = no_network
        self.auto_allowed: Set[Effect] = set(auto_allowed or AUTO_ALLOWED_DEFAULT)
        self.prompt_fn = prompt_fn
        self.session_allowed: Set[Effect] = set()

    def decide(self, effect: Effect, name: str, description: str) -> Decision:
        if effect == "network" and self.no_network:
            return Decision(False, "network disabled (--no-network)")
        if effect in self.auto_allowed or effect in self.session_allowed:
            return Decision(True, "auto-allowed")
        if effect not in GATED_EFFECTS:
            return Decision(True, "auto-allowed")

        if self.mode == "yes":
            return Decision(True, "auto-approved (--yes)")
        if self.mode == "dry_run":
            return Decision(True, "dry-run (not executed)", simulated=True)

        # Interactive.
        if self.prompt_fn is None:
            return Decision(False, "no approver available")
        answer = (self.prompt_fn(effect, name, description) or "").strip().lower()
        if answer == APPROVE_SESSION:
            self.session_allowed.add(effect)
            return Decision(True, "approved for session")
        if answer == APPROVE_ONCE:
            return Decision(True, "approved")
        return Decision(False, "denied by user")


class AuditLog:
    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)


def clamp_output(text: str, limit: int) -> str:
    """Truncate long tool output, leaving a marker with the elided line count."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    remaining = text[limit:]
    elided_lines = remaining.count("\n") + 1
    return f"{head}\n... [truncated {len(remaining)} chars / ~{elided_lines} lines]"
