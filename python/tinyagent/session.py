"""Conversation session state with optional local save/resume."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]


@dataclass
class Session:
    system_prompt: str
    messages: List[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: Optional[List[Message]] = None) -> None:
        msg: Message = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "name": name, "content": content})

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"system_prompt": self.system_prompt, "messages": self.messages}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        session = cls(system_prompt=data["system_prompt"], messages=data["messages"])
        return session
