"""Persistent local TinyEngine daemon for reusing a loaded model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from tinyengine.runtime import TinyEngineError


class EngineLike(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...


def default_socket_path(model_path: str, context_tokens: int) -> Path:
    override = os.environ.get("TINYAGENT_DAEMON_SOCKET")
    if override:
        return Path(override)
    resolved = str(Path(model_path).expanduser().resolve())
    key = hashlib.sha256(f"{resolved}|ctx={context_tokens}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"tinyagent-{os.getuid()}-{key}.sock"


def _json_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _connect(socket_path: Path, timeout: float | None = None) -> socket.socket:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        conn.settimeout(timeout)
    conn.connect(str(socket_path))
    conn.settimeout(None)
    return conn


def _request(socket_path: Path, request: dict[str, Any], on_event: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
    try:
        conn = _connect(socket_path)
    except OSError as exc:
        raise TinyEngineError(f"daemon unavailable: {exc}") from exc
    with conn:
        reader = conn.makefile("rb")
        writer = conn.makefile("wb")
        writer.write(_json_line(request))
        writer.flush()
        while True:
            line = reader.readline()
            if not line:
                raise TinyEngineError("daemon closed connection")
            event = json.loads(line.decode())
            if "token" in event:
                if on_event is not None:
                    on_event(event)
                continue
            if event.get("ok"):
                return event
            if "error" in event:
                raise TinyEngineError(str(event["error"]))
            raise TinyEngineError(f"malformed daemon response: {event!r}")


def _daemon_ready(socket_path: Path) -> bool:
    try:
        response = _request(socket_path, {"op": "ping"})
    except TinyEngineError:
        return False
    return bool(response.get("ok"))


def _start_daemon(socket_path: Path, model_path: str, context_tokens: int, startup_timeout: float) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError:
            pass
    log_path = socket_path.with_suffix(".log")
    log = log_path.open("ab")
    cmd = [
        sys.executable,
        "-m",
        "tinyagent.daemon",
        "--socket",
        str(socket_path),
        "--model",
        model_path,
        "--ctx",
        str(context_tokens),
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _daemon_ready(socket_path):
            return
        if process.poll() is not None:
            raise TinyEngineError(f"daemon exited during startup; see {log_path}")
        time.sleep(0.1)
    raise TinyEngineError(f"daemon did not become ready within {startup_timeout:.0f}s; see {log_path}")


class DaemonEngine:
    """Engine-compatible client that talks to a persistent TinyEngine daemon."""

    def __init__(
        self,
        model_path: str,
        context_tokens: int,
        socket_path: Optional[Path] = None,
        auto_start: bool = True,
        startup_timeout: float = 30.0,
    ) -> None:
        self.model_path = model_path
        self.context_tokens = context_tokens
        self.socket_path = socket_path or default_socket_path(model_path, context_tokens)
        if auto_start and not _daemon_ready(self.socket_path):
            _start_daemon(self.socket_path, model_path, context_tokens, startup_timeout)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        chunks: list[str] = []

        def handle(event: dict[str, Any]) -> None:
            chunk = str(event.get("token", ""))
            token_id = int(event.get("token_id", 0))
            chunks.append(chunk)
            if on_token is not None:
                on_token(chunk, token_id)

        response = _request(
            self.socket_path,
            {"op": "generate", "prompt": prompt, "max_tokens": max_tokens},
            on_event=handle,
        )
        text = str(response.get("text", ""))
        return text if not chunks else "".join(chunks)

    def count_tokens(self, text: str) -> int:
        response = _request(self.socket_path, {"op": "count_tokens", "text": text})
        return int(response["count"])

    def shutdown(self) -> None:
        _request(self.socket_path, {"op": "shutdown"})


def serve_engine(engine: EngineLike, socket_path: Path, max_requests: int | None = None) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(8)
        handled = 0
        running = True
        while running and (max_requests is None or handled < max_requests):
            conn, _ = server.accept()
            handled += 1
            with conn:
                reader = conn.makefile("rb")
                writer = conn.makefile("wb")
                running = _handle_connection(engine, reader, writer)
    finally:
        server.close()
        try:
            socket_path.unlink()
        except OSError:
            pass


def _handle_connection(engine: EngineLike, reader, writer) -> bool:
    line = reader.readline()
    if not line:
        return True
    try:
        request = json.loads(line.decode())
        op = request.get("op")
        if op == "ping":
            writer.write(_json_line({"ok": True}))
        elif op == "count_tokens":
            writer.write(_json_line({"ok": True, "count": engine.count_tokens(str(request.get("text", "")))}))
        elif op == "generate":
            chunks: list[str] = []

            def on_token(chunk: str, token_id: int) -> None:
                chunks.append(chunk)
                writer.write(_json_line({"token": chunk, "token_id": token_id}))
                writer.flush()

            text = engine.generate(str(request.get("prompt", "")), int(request.get("max_tokens", 0)), on_token)
            writer.write(_json_line({"ok": True, "text": text if not chunks else "".join(chunks)}))
        elif op == "shutdown":
            writer.write(_json_line({"ok": True}))
            writer.flush()
            return False
        else:
            writer.write(_json_line({"error": f"unknown daemon op: {op}"}))
    except Exception as exc:  # noqa: BLE001 - protocol boundary reports errors to client
        writer.write(_json_line({"error": f"{type(exc).__name__}: {exc}"}))
    writer.flush()
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a persistent TinyAgent model daemon.")
    parser.add_argument("--socket", required=True, help="Unix socket path.")
    parser.add_argument("--model", required=True, help="Path to local Qwen GGUF.")
    parser.add_argument("--ctx", type=int, default=512, help="Model context tokens.")
    args = parser.parse_args(argv)

    from .engine import TinyEngine

    engine = TinyEngine(args.model, context_tokens=args.ctx)
    serve_engine(engine, Path(args.socket))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
