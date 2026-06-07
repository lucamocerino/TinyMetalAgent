"""The ReAct agent loop: render -> generate -> parse -> approve -> execute -> repeat."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .context import compact_history, trim_to_budget
from .engine import Engine
from .safety import AuditEntry, AuditLog, Approver, clamp_output
from .session import Session
from .template import render_chat
from .toolcall import ToolCall, parse_tool_calls
from .tools import ToolContext, ToolRegistry
from .tools.files import ToolError
from tinyengine.runtime import TinyEngineError

DEFAULT_SYSTEM_PROMPT = (
    "You are tinyagent, a local coding assistant.\n"
    "CHAT: answer conversational questions directly, without tools.\n"
    "ACT: for file/code/shell tasks, emit exactly one "
    "<tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>.\n"
    "Use write_file/apply_patch for edits, read/list/grep before inspecting, "
    "run_shell for tests, and finish when complete."
)

_NUDGE_NO_CALL = (
    "You did not call a tool. This task requires creating/modifying files or running "
    "commands, so do not answer in prose or markdown. Emit exactly one <tool_call>"
    "{\"name\": ..., \"arguments\": {...}}</tool_call> (e.g. write_file / apply_patch / "
    "run_shell), or call the finish tool if the task is already complete."
)

_NUDGE_MALFORMED = (
    "Your previous tool call was malformed and could not be parsed. Re-emit it as a single "
    "valid <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> JSON object, with no "
    "extra commentary."
)


def _looks_like_unstructured_action(text: str) -> bool:
    """Heuristic: did the model dump code/files in prose instead of calling a tool?

    A fenced code block is the strong signal we observed: stronger models sometimes
    answer a "create file / write tests" task with a markdown tutorial rather than a
    ``<tool_call>``. Plain prose answers (no fence) are left alone so genuine Q&A
    replies still terminate the loop.
    """
    return "```" in text or "~~~" in text


_QUESTION_CUES = (
    "explain", "what", "why", "how", "describe", "tell me", "difference",
    "who", "when", "where", "which", "should i", "can you explain", "help me understand",
)

_CHAT_CUES = (
    "hi", "hey", "hello", "ciao", "salve", "buongiorno", "buonasera",
    "thanks", "thank you", "grazie", "ok", "okay",
)

_ACTION_CUES = (
    "add", "apply", "build", "change", "check", "commit", "create", "delete",
    "edit", "fix", "grep", "implement", "inspect", "install", "list", "make",
    "modify", "move", "patch", "read", "refactor", "remove", "rename", "run",
    "search", "test", "tests", "unittest", "unittests", "unit", "update", "write",
)


def _looks_like_question(text: str) -> bool:
    """Heuristic: is the user asking/chatting rather than requesting file work?

    When the user asks a question, an explanatory answer (even one containing a
    markdown code snippet) is a legitimate, informative reply and must NOT be
    nudged into a file write. Action verbs like "create"/"write"/"fix" are not
    treated as questions, so build tasks still get the tool-protocol nudge.
    """
    stripped = text.strip().lower()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first = stripped.split()[0] if stripped.split() else ""
    return first in _QUESTION_CUES or any(
        stripped.startswith(cue + " ") or f" {cue} " in stripped for cue in _QUESTION_CUES
    )


def _looks_like_action_request(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    words = stripped.replace("/", " ").replace("-", " ").split()
    first = words[0] if words else ""
    if first in _ACTION_CUES:
        return True
    action_phrases = (
        "can we create", "can you create", "please create",
        "can we add", "can you add", "please add",
        "can we write", "can you write", "please write",
        "can we make", "can you make", "please make",
        "can we run", "can you run", "please run",
        "can we test", "can you test", "please test",
    )
    return any(stripped.startswith(phrase) for phrase in action_phrases)


def _looks_like_chat(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    if _looks_like_question(stripped) and not _looks_like_action_request(stripped):
        return True
    return (
        stripped in _CHAT_CUES
        or any(stripped.startswith(cue + " ") for cue in _CHAT_CUES)
    )


@dataclass
class AgentEvent:
    """Emitted to the UI as the loop progresses."""

    kind: str  # status | assistant_text | tool_call | tool_result | final | step
    text: str = ""
    name: str = ""
    arguments: Optional[Dict[str, Any]] = None
    allowed: bool = True
    step: int = 0
    elapsed_ms: float = 0.0


EventSink = Callable[[AgentEvent], None]


@dataclass
class AgentResult:
    answer: str
    steps: int
    finished: bool


class Agent:
    def __init__(
        self,
        engine: Engine,
        registry: ToolRegistry,
        approver: Approver,
        tool_context: ToolContext,
        session: Session,
        max_steps: int = 12,
        max_new_tokens: int = 384,
        token_budget: int = 1600,
        no_tool_retries: int = 1,
        on_event: Optional[EventSink] = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.approver = approver
        self.tool_context = tool_context
        self.session = session
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.token_budget = token_budget
        self.no_tool_retries = no_tool_retries
        self.on_event = on_event
        self.audit = AuditLog()
        self._tool_schemas_cache: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}

    # -- helpers ---------------------------------------------------------
    def _emit(self, event: AgentEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _tool_schemas(self, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        key = tuple(names or self.registry.names())
        if key not in self._tool_schemas_cache:
            self._tool_schemas_cache[key] = self.registry.schemas(compact=True, names=key)
        return self._tool_schemas_cache[key]

    def _render(self, messages: List[Dict[str, Any]], tool_names: Optional[Sequence[str]] = None) -> str:
        return render_chat(messages, self._tool_schemas(tool_names), add_generation_prompt=True)

    def _fit_messages(
        self,
        include_tools: bool = True,
        tool_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        tools = self._tool_schemas(tool_names) if include_tools else None
        render_no_gen = lambda m: render_chat(m, tools, add_generation_prompt=False)
        rendered = render_no_gen(self.session.messages)
        if self.engine.count_tokens(rendered) <= self.token_budget:
            return list(self.session.messages)

        trimmed = trim_to_budget(
            self.session.messages,
            render_no_gen,
            self.engine.count_tokens,
            self.token_budget,
            keep_recent=4 if include_tools else 6,
            min_recent=2,
        )
        if self.engine.count_tokens(render_no_gen(trimmed)) <= self.token_budget:
            return trimmed

        self.session.messages = compact_history(
            self.engine, self.session.messages, render_fn=render_no_gen
        )
        return trim_to_budget(
            self.session.messages, render_no_gen, self.engine.count_tokens, self.token_budget
        )

    def _select_tool_names(self, user_message: str, tools_ran: bool = False) -> List[str]:
        text = user_message.lower()
        words = set(text.replace("/", " ").replace("-", " ").replace("_", " ").split())
        names: List[str]
        if any(word in words for word in ("test", "tests", "unittest", "unittests")) or "unit test" in text:
            names = ["glob", "read_file", "write_file", "run_shell"]
        elif any(word in words for word in ("create", "write", "add", "make")) or "new folder" in text or "new file" in text:
            names = ["write_file", "list_dir", "run_shell"]
        elif any(word in words for word in ("grep", "search", "find")):
            names = ["grep", "glob", "read_file"]
        elif any(word in words for word in ("run", "test", "build", "install", "shell", "command")):
            names = ["run_shell", "read_file"]
        elif any(word in words for word in ("edit", "modify", "patch", "refactor", "fix", "change")):
            names = ["read_file", "apply_patch", "write_file"]
        elif tools_ran:
            names = ["read_file", "write_file", "apply_patch", "run_shell"]
        else:
            names = ["read_file", "list_dir", "write_file"]
        if "finish" not in names:
            names.append("finish")
        return [name for name in names if name in self.registry]

    def _execute(self, call: ToolCall, step: int) -> str:
        spec = self.registry.get(call.name)
        if spec is None:
            return f"error: unknown tool '{call.name}'"
        description = f"{call.name}({call.arguments})"
        decision = self.approver.decide(spec.effect, call.name, description)
        if not decision.allowed:
            self.audit.record(AuditEntry(step, call.name, spec.effect, call.arguments, False, False, decision.reason))
            return f"[denied] {decision.reason}"
        if decision.simulated:
            self.audit.record(AuditEntry(step, call.name, spec.effect, call.arguments, True, True, "dry-run"))
            return f"[dry-run] {spec.effect} tool '{call.name}' was not executed"
        try:
            result = spec.handler(call.arguments, self.tool_context)
        except ToolError as exc:
            result = f"error: {exc}"
        except Exception as exc:  # noqa: BLE001 - report any tool failure to the model
            result = f"error: {type(exc).__name__}: {exc}"
        result = clamp_output(result, self.tool_context.limits.max_output_chars)
        self.audit.record(AuditEntry(step, call.name, spec.effect, call.arguments, True, False, result[:120]))
        return result

    # -- main loop -------------------------------------------------------
    def run(self, user_message: str) -> AgentResult:
        if _looks_like_chat(user_message):
            return self.chat(user_message)

        self.session.add_user(user_message)
        last_text = ""
        nudges_used = 0
        tools_ran = False
        user_is_asking = _looks_like_question(user_message)
        user_wants_action = _looks_like_action_request(user_message)
        tool_names = self._select_tool_names(user_message)
        for step in range(1, self.max_steps + 1):
            self._emit(AgentEvent(kind="step", step=step))
            messages = self._fit_messages(tool_names=tool_names)
            prompt = self._render(messages, tool_names=tool_names)
            self._emit(AgentEvent(
                kind="status",
                text=(
                    f"rendered ACT prompt ({len(messages)} messages, "
                    f"tools {','.join(tool_names)}, budget {self.token_budget} tokens)"
                ),
                step=step,
            ))
            started = time.perf_counter()
            try:
                text = self.engine.generate(prompt, self.max_new_tokens)
            except TinyEngineError as exc:
                message = f"model generation failed: {exc}"
                self._emit(AgentEvent(kind="final", text=message, step=step))
                return AgentResult(answer=message, steps=step, finished=False)
            self._emit(AgentEvent(
                kind="status",
                text="model response received",
                step=step,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            ))
            parsed = parse_tool_calls(text)
            last_text = parsed.content or text

            if not parsed.has_calls:
                # The model replied without a tool call. If it dumped code/files in
                # markdown (before doing any real work) or emitted a malformed tool
                # call, nudge it back onto the tool protocol instead of silently
                # treating the prose as the final answer. When the user is plainly
                # asking a question, an explanatory (possibly fenced) answer is the
                # desired output, so we accept it as final without nudging.
                malformed = bool(parsed.malformed)
                wants_action = malformed or (
                    user_wants_action
                    and not tools_ran
                ) or (
                    _looks_like_unstructured_action(text)
                    and not tools_ran
                    and not user_is_asking
                )
                if wants_action and nudges_used < self.no_tool_retries:
                    self.session.add_assistant(text)
                    self.session.add_user(_NUDGE_MALFORMED if malformed else _NUDGE_NO_CALL)
                    nudges_used += 1
                    continue
                self.session.add_assistant(text)
                if user_wants_action and not tools_ran:
                    message = "I could not produce a valid tool call for this task."
                    self._emit(AgentEvent(kind="final", text=message, step=step))
                    return AgentResult(answer=message, steps=step, finished=False)
                self._emit(AgentEvent(kind="final", text=last_text, step=step))
                return AgentResult(answer=last_text, steps=step, finished=True)

            tool_call_payload = [{"name": c.name, "arguments": c.arguments} for c in parsed.tool_calls]
            self.session.add_assistant(parsed.content, tool_calls=tool_call_payload)
            if parsed.content:
                self._emit(AgentEvent(kind="assistant_text", text=parsed.content, step=step))

            for call in parsed.tool_calls:
                if call.name == "finish":
                    answer = str(call.arguments.get("answer", "")) or parsed.content
                    self._emit(AgentEvent(kind="final", text=answer, step=step))
                    return AgentResult(answer=answer, steps=step, finished=True)

                spec = self.registry.get(call.name)
                effect = spec.effect if spec else "read"
                self._emit(AgentEvent(kind="tool_call", name=call.name, arguments=call.arguments, step=step))
                tool_started = time.perf_counter()
                result = self._execute(call, step)
                self.session.add_tool_result(call.name, result)
                self._emit(AgentEvent(
                    kind="tool_result",
                    name=call.name,
                    text=result,
                    step=step,
                    elapsed_ms=(time.perf_counter() - tool_started) * 1000.0,
                ))
                tools_ran = True
                nudges_used = 0
                tool_names = self._select_tool_names(user_message, tools_ran=True)

        return AgentResult(answer=last_text, steps=self.max_steps, finished=False)

    # -- ask / chat mode -------------------------------------------------
    def chat(self, user_message: str, on_token: Optional[Callable[[str, int], None]] = None) -> AgentResult:
        """Single-turn conversational reply with tools disabled (``/ask`` mode).

        The prompt is rendered without tool schemas so the model answers the
        question directly in prose instead of attempting tool calls. Any stray
        ``<tool_call>`` markup the model emits anyway is stripped, leaving only
        the explanatory text. ``on_token`` streams raw chunks as they decode.
        """
        self.session.add_user(user_message)
        messages = self._fit_messages(include_tools=False)
        prompt = render_chat(messages, tools=None, add_generation_prompt=True)
        self._emit(AgentEvent(
            kind="status",
            text=f"rendered chat prompt ({len(messages)} messages, no tools)",
            step=1,
        ))
        started = time.perf_counter()
        try:
            text = self.engine.generate(prompt, self.max_new_tokens, on_token)
        except TinyEngineError as exc:
            message = f"model generation failed: {exc}"
            self.session.add_assistant(message)
            self._emit(AgentEvent(kind="final", text=message, step=1))
            return AgentResult(answer=message, steps=1, finished=False)
        self._emit(AgentEvent(
            kind="status",
            text="chat response received",
            step=1,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        ))
        parsed = parse_tool_calls(text)
        answer = (parsed.content or text).strip()
        self.session.add_assistant(answer)
        self._emit(AgentEvent(kind="final", text=answer, step=1))
        return AgentResult(answer=answer, steps=1, finished=True)
