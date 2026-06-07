"""Incremental display filter for streamed model output.

The engine streams raw model text token-by-token. In chat/ask mode the model
should produce pure prose, but a small local model occasionally emits stray
``<tool_call>...</tool_call>`` markup anyway. ``StreamFilter`` lets the CLI print
tokens live while hiding those spans, correctly handling tags that are split
across chunk boundaries (e.g. ``"<tool"`` in one chunk, ``"_call>"`` in the next).
"""
from __future__ import annotations

from typing import Callable

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"


def _longest_prefix_held(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a non-empty prefix of ``tag``.

    Those trailing chars must be withheld: they might grow into ``tag`` on the next
    chunk, so emitting them now could leak the start of a tag we mean to hide.
    """
    limit = min(len(buf), len(tag) - 1)
    for size in range(limit, 0, -1):
        if tag.startswith(buf[-size:]):
            return size
    return 0


class StreamFilter:
    """Feed model chunks in; emit display-safe text with tool-call spans removed."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buf = ""
        self._suppressing = False

    def feed(self, chunk: str, _token_id: int = 0) -> None:
        self._buf += chunk
        self._drain(flush=False)

    def close(self) -> None:
        """Flush any remaining buffered text (call once the stream ends)."""
        self._drain(flush=True)
        if self._buf and not self._suppressing:
            self._emit(self._buf)
        self._buf = ""

    def _drain(self, flush: bool) -> None:
        while self._buf:
            if not self._suppressing:
                idx = self._buf.find(_OPEN)
                if idx != -1:
                    if idx:
                        self._emit(self._buf[:idx])
                    self._buf = self._buf[idx + len(_OPEN):]
                    self._suppressing = True
                    continue
                if flush:
                    return
                hold = _longest_prefix_held(self._buf, _OPEN)
                if hold:
                    self._emit(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    self._emit(self._buf)
                    self._buf = ""
                return
            else:
                idx = self._buf.find(_CLOSE)
                if idx != -1:
                    self._buf = self._buf[idx + len(_CLOSE):]
                    self._suppressing = False
                    continue
                if flush:
                    self._buf = ""
                    return
                hold = _longest_prefix_held(self._buf, _CLOSE)
                self._buf = self._buf[-hold:] if hold else ""
                return
