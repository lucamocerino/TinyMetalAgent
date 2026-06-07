"""tinyagent command-line client: one-shot and interactive REPL."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from .agent import Agent, AgentEvent, DEFAULT_SYSTEM_PROMPT
from .engine import TinyEngine
from .safety import Approver, Limits
from .session import Session
from .template import render_chat
from .tools import ToolContext, default_registry
from .tools.files import ToolError, resolve_path
from .tools.shell import run_shell
from tinyengine.runtime import detect_arch

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Searched in order when --model is omitted. Paths are resolved both relative to the
# current directory and to the repo root, so `tinyagent` works from anywhere.
_DEFAULT_MODEL_NAMES = [
    "../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf",
    "models/qwen2.5-coder-3b-instruct-q4_0-te.gguf",
]

_SLASH_COMMANDS = (
    "/help",
    "/ask",
    "/tools",
    "/context",
    "/diff",
    "/clear",
    "/compact",
    "/cwd",
    "/approve auto",
    "/approve ask",
    "/save",
    "/exit",
    "/quit",
)

_QUICK_HELP = """\
Quick help:
  /        list commands
  ?        show this quick help
  @file    attach a file to the next prompt
  !cmd     run a shell command in the workspace
  Tab      autocomplete slash commands and @file paths
  /help    full command help"""

_MENTION_RE = re.compile(r"(^|\s)@([^\s]+)")


def _model_candidates(explicit: Optional[str]) -> List[str]:
    if explicit:
        return [explicit]
    candidates: List[str] = []
    env = os.environ.get("TINYAGENT_MODEL", "")
    if env:
        candidates.append(env)
    for name in _DEFAULT_MODEL_NAMES:
        candidates.append(name)                       # relative to cwd
        candidates.append(str((_REPO_ROOT / name)))   # relative to repo root
    return candidates


def _find_model(explicit: Optional[str]) -> str:
    for cand in _model_candidates(explicit):
        if cand and Path(cand).is_file():
            return str(Path(cand).resolve())
    raise SystemExit(
        "no local Qwen GGUF found. Pass --model <path.gguf> or set TINYAGENT_MODEL."
    )


def _default_context_tokens() -> int:
    try:
        recommended = detect_arch().recommended_max_context
    except Exception:
        return 512
    return recommended or 512


def _resolve_context_tokens(value: Optional[int]) -> int:
    return value if value is not None else _default_context_tokens()


def _resolve_max_new_tokens(value: Optional[int], context_tokens: int) -> int:
    if value is not None:
        return value
    return 128 if context_tokens <= 512 else 384


# -- terminal rendering --------------------------------------------------
class Renderer:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.streaming = False

    def __call__(self, event: AgentEvent) -> None:
        if event.kind == "status":
            if not self.verbose:
                return
            suffix = f" ({event.elapsed_ms:.0f} ms)" if event.elapsed_ms else ""
            prefix = "\033[0m\n" if self.streaming else ""
            print(f"{prefix}\033[90m• {event.text}{suffix}\033[0m", file=sys.stderr)
            if self.streaming:
                sys.stdout.write("\033[36m")
                sys.stdout.flush()
        elif event.kind == "step":
            if self.verbose:
                print(f"\n\033[90m— step {event.step} —\033[0m", file=sys.stderr)
        elif event.kind == "assistant_text" and event.text.strip():
            print(f"\033[36m{event.text.strip()}\033[0m")
        elif event.kind == "tool_call":
            args = f"({_fmt_args(event.arguments)})" if self.verbose else ""
            print(f"\033[33m→ tool {event.name}{args}\033[0m")
        elif event.kind == "tool_result":
            if self.verbose:
                body = event.text
                suffix = f" ({event.elapsed_ms:.0f} ms)" if event.elapsed_ms else ""
                print(f"\033[90m✓ {event.name}{suffix}\n{body}\033[0m")
            else:
                print(f"\033[90m✓ {event.name}\033[0m")
        elif event.kind == "final":
            # When streaming, the answer was already printed live token-by-token.
            if self.streaming:
                return
            print(f"\033[1;32m{event.text.strip()}\033[0m")


def stream_chat(agent: Agent, renderer: Renderer, message: str) -> None:
    """Run a chat-mode turn, printing tokens live as the model decodes them."""
    from .streaming import StreamFilter

    sys.stdout.write("\033[36m")
    sys.stdout.flush()

    def emit(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    flt = StreamFilter(emit)
    renderer.streaming = True
    try:
        agent.chat(message, on_token=flt.feed)
    finally:
        flt.close()
        renderer.streaming = False
        sys.stdout.write("\033[0m\n")
        sys.stdout.flush()


def _fmt_args(args: Optional[dict]) -> str:
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


def _short(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …"


def _approval_prompt(effect: str, name: str, description: str) -> str:
    try:
        return input(f"\033[1;31mApprove {effect} tool {name}? [y/N/a=allow-all] \033[0m")
    except EOFError:
        return "n"


def _slash_matches(line: str) -> List[str]:
    if not line.startswith("/") or " " in line:
        return []
    return [cmd for cmd in _SLASH_COMMANDS if cmd.startswith(line)]


def _mention_matches(text: str, root: Path) -> List[str]:
    if not text.startswith("@"):
        return []
    raw = text[1:]
    base_raw = str(Path(raw).parent)
    name_prefix = Path(raw).name
    base_rel = "" if base_raw == "." else base_raw
    base = (root / base_rel).resolve()
    try:
        base.relative_to(root.resolve())
    except ValueError:
        return []
    if not base.is_dir():
        return []
    matches: List[str] = []
    for child in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if not child.name.startswith(name_prefix):
            continue
        rel = child.resolve().relative_to(root.resolve())
        suffix = "/" if child.is_dir() else ""
        matches.append("@" + str(rel) + suffix)
    return matches


def _install_repl_completion(root: Path) -> None:
    try:
        import readline
    except ImportError:
        return

    def complete(text: str, state: int) -> Optional[str]:
        line = readline.get_line_buffer()
        matches = _slash_matches(line) if line.startswith("/") else _mention_matches(text, root)
        return matches[state] if state < len(matches) else None

    try:
        readline.set_completer(complete)
        readline.set_completer_delims(readline.get_completer_delims().replace("/", ""))
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("bind ^I rl_complete")
    except Exception:
        return


def _expand_mentions(line: str, ctx: ToolContext, max_chars: int = 4000) -> str:
    mentions = []
    for match in _MENTION_RE.finditer(line):
        raw = match.group(2)
        try:
            path = resolve_path(ctx, raw, must_exist=True)
        except ToolError as exc:
            mentions.append(f"--- @{raw} ---\n[mention error: {exc}]")
            continue
        if path.is_dir():
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
            body = "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries)
        else:
            body = path.read_text(encoding="utf-8", errors="replace")
        if len(body) > max_chars:
            body = body[:max_chars] + "\n... [truncated]"
        mentions.append(f"--- @{raw} ---\n{body}")
    if not mentions:
        return line
    return line + "\n\nAttached context:\n" + "\n\n".join(mentions)


def _context_report(agent: Agent) -> str:
    messages = agent.session.messages
    context_tokens = getattr(agent.engine, "context_tokens", 0) or 0
    chat_prompt = render_chat(messages, tools=None, add_generation_prompt=True)
    chat_tokens = agent.engine.count_tokens(chat_prompt)
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    tool_names = agent._select_tool_names(str(last_user))
    act_prompt = agent._render(agent._fit_messages(tool_names=tool_names), tool_names=tool_names)
    act_tokens = agent.engine.count_tokens(act_prompt)
    limit = f"/{context_tokens}" if context_tokens else ""
    return (
        f"messages: {len(messages)}\n"
        f"chat prompt: {chat_tokens}{limit} tokens\n"
        f"ACT prompt: {act_tokens}{limit} tokens\n"
        f"ACT tools: {', '.join(tool_names)}\n"
        f"reply budget: {agent.max_new_tokens} tokens"
    )


def _run_shell_escape(command: str, agent: Agent, renderer: Renderer) -> int:
    if not command.strip():
        print("usage: !<shell command>")
        return 1
    try:
        output = run_shell({"cmd": command}, agent.tool_context)
    except ToolError as exc:
        print(f"error: {exc}")
        return 1
    print(output if renderer.verbose else _short(output, limit=1200))
    return 0


# -- agent construction --------------------------------------------------
def build_agent(args: argparse.Namespace, session: Session, on_event) -> Agent:
    model_path = _find_model(args.model)
    context_tokens = _resolve_context_tokens(args.ctx)
    max_new_tokens = _resolve_max_new_tokens(args.max_new_tokens, context_tokens)
    if on_event is not None:
        mode = "daemon" if args.daemon else "direct"
        on_event(AgentEvent(
            kind="status",
            text=f"loading model ({mode}, ctx={context_tokens}, max_new={max_new_tokens})",
        ))
    started = time.perf_counter()
    if args.daemon:
        from .daemon import DaemonEngine

        engine = DaemonEngine(model_path, context_tokens=context_tokens)
    else:
        engine = TinyEngine(model_path, context_tokens=context_tokens)
    if on_event is not None:
        on_event(AgentEvent(
            kind="status",
            text=f"model ready: {Path(model_path).name}",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        ))
    registry = default_registry(include_network=not args.no_network)
    mode = "yes" if args.yes else "dry_run" if args.dry_run else "ask"
    approver = Approver(mode=mode, no_network=args.no_network, prompt_fn=_approval_prompt)
    tool_context = ToolContext(
        root=Path(args.root).resolve(),
        limits=Limits(shell_timeout=args.shell_timeout, max_steps=args.max_steps),
    )
    return Agent(
        engine=engine,
        registry=registry,
        approver=approver,
        tool_context=tool_context,
        session=session,
        max_steps=args.max_steps,
        max_new_tokens=max_new_tokens,
        token_budget=max(32, context_tokens - max_new_tokens - 64),
        on_event=on_event,
    )


# -- REPL ----------------------------------------------------------------
_HELP = """\
Commands:
  /help            show this help
  /ask [question]  with text: answer one question in chat mode (no tools);
                   with no text: toggle persistent chat-only mode
  /tools           list available tools
  /context         show context/token usage
  /diff            show git diff summary
  /clear           reset the conversation
  /compact         summarize history to free context
  /cwd             show the workspace root
  /approve auto    auto-approve all tools this session
  /approve ask     require approval for write/exec tools
  /save <file>     save the conversation
  /exit            quit
Other input:
  ?                quick help
  @file            attach file content to the prompt
  !cmd             run a shell command in the workspace
Anything else runs the agent (it may use tools). In chat-only mode it is
answered conversationally without tools."""


def run_repl(agent: Agent, args: argparse.Namespace, renderer: "Renderer") -> int:
    from .context import compact_history

    _install_repl_completion(agent.tool_context.root)
    print("tinyagent — local Qwen coding agent. /help for commands, /exit to quit.")
    model_path = getattr(agent.engine, "model_path", "?")
    print(f"\033[90mmodel: {model_path}  root: {agent.tool_context.root}\033[0m")
    print("\033[90mAsk questions for explanations, or describe a task to run the agent. "
          "Use /ask for chat-only.\033[0m")
    ask_mode = bool(getattr(args, "ask", False))
    while True:
        prompt_label = "tinyagent(ask)>" if ask_mode else "tinyagent>"
        try:
            line = input(f"\033[1;34m{prompt_label}\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "?":
            print(_QUICK_HELP)
            continue
        if line.startswith("!"):
            _run_shell_escape(line[1:].strip(), agent, renderer)
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if not cmd:
                print("Commands: " + ", ".join(_SLASH_COMMANDS))
                continue
            if cmd in ("exit", "quit"):
                return 0
            if cmd == "help":
                print(_HELP)
            elif cmd == "ask":
                question = rest.strip()
                if question:
                    stream_chat(agent, renderer, question)
                else:
                    ask_mode = not ask_mode
                    print(f"(chat-only mode {'on' if ask_mode else 'off'})")
            elif cmd == "tools":
                print("\n".join(f"  {s.name} [{s.effect}] — {s.description}" for s in agent.registry))
            elif cmd == "context":
                print(_context_report(agent))
            elif cmd == "diff":
                _run_shell_escape("git --no-pager diff --stat", agent, renderer)
            elif cmd == "clear":
                agent.session.reset()
                print("(conversation cleared)")
            elif cmd == "compact":
                agent.session.messages = compact_history(agent.engine, agent.session.messages)
                print("(history compacted)")
            elif cmd == "cwd":
                print(agent.tool_context.root)
            elif cmd == "approve":
                if rest.strip() == "auto":
                    agent.approver.mode = "yes"
                    print("(auto-approving all tools)")
                else:
                    agent.approver.mode = "ask"
                    agent.approver.session_allowed.clear()
                    print("(will ask for write/exec tools)")
            elif cmd == "save":
                target = rest.strip() or "tinyagent-session.json"
                agent.session.save(target)
                print(f"(saved to {target})")
            else:
                print(f"unknown command: /{cmd}")
            continue
        line = _expand_mentions(line, agent.tool_context)
        if ask_mode:
            stream_chat(agent, renderer, line)
        else:
            agent.run(line)
    return 0


# -- entry point ---------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyagent",
        description="Local-model terminal coding agent (Qwen via TinyEngine).",
    )
    parser.add_argument("task", nargs="*", help="Task to run (omit for interactive REPL).")
    parser.add_argument("--model", help="Path to a local Qwen GGUF.")
    parser.add_argument("--root", default=".", help="Workspace root (default: cwd).")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum generated tokens (default: 128 for 512 ctx, otherwise 384).")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Model context tokens (default: detected hardware recommendation).")
    parser.add_argument("--daemon", action="store_true", help="Reuse/start a persistent local model daemon.")
    parser.add_argument("--shell-timeout", type=float, default=30.0)
    parser.add_argument("--yes", action="store_true", help="Auto-approve all tools.")
    parser.add_argument("--ask", action="store_true",
                        help="Chat-only mode: answer conversationally, never use tools.")
    parser.add_argument("--dry-run", action="store_true", help="Never execute write/exec tools.")
    parser.add_argument("--no-network", action="store_true", help="Disable network tools.")
    parser.add_argument("--verbose", action="store_true", help="Show steps and full tool output.")
    parser.add_argument("--system", help="Override the system prompt.")
    parser.add_argument("--resume", help="Resume a saved session JSON.")
    parser.add_argument("--save", help="Save the session to this file when done.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    system_prompt = args.system or DEFAULT_SYSTEM_PROMPT
    session = Session.load(args.resume) if args.resume else Session(system_prompt=system_prompt)
    renderer = Renderer(verbose=args.verbose)
    agent = build_agent(args, session, renderer)

    try:
        if args.task:
            task_text = " ".join(args.task)
            task_text = _expand_mentions(task_text, agent.tool_context)
            if args.ask:
                stream_chat(agent, renderer, task_text)
                code = 0
            else:
                result = agent.run(task_text)
                code = 0 if result.finished else 1
        else:
            code = run_repl(agent, args, renderer)
    finally:
        if args.save:
            agent.session.save(args.save)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
