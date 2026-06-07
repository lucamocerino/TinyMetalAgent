"""Adapter around the local TinyEngine model.

Defines the minimal ``Engine`` surface the agent needs (``generate`` +
``count_tokens``) so the agent loop can be unit-tested with a fake engine and
run for real against the local Qwen GGUF.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from tinyengine.runtime import TinyEngineError


class Engine(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...


class TinyEngine:
    """Backs the agent with a local Qwen GGUF via TinyEngine (no remote calls)."""

    def __init__(self, model_path: str, context_tokens: int = 2048) -> None:
        from tinyengine.runtime import Model, RuntimeOptions

        self.model_path = model_path
        self.context_tokens = context_tokens
        self._model = Model(model_path, RuntimeOptions(context_tokens=context_tokens))

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        try:
            return self._model.generate_raw(prompt, max_tokens, on_token)
        except TinyEngineError as exc:
            if str(exc) != "unsupported":
                raise

        # Avoid counting tokens on the fast path: the C runtime tokenizes again for
        # generation. Only retry with a capped length when the runtime rejected a
        # request that exceeds the pre-allocated KV context.
        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens + max_tokens + 8 <= self.context_tokens:
            raise TinyEngineError("unsupported")
        room = self.context_tokens - prompt_tokens - 8
        if room < 1:
            raise TinyEngineError("prompt exceeds context window")
        capped = max_tokens if max_tokens < room else room
        try:
            return self._model.generate_raw(prompt, capped, on_token)
        except TinyEngineError as exc:
            if str(exc) == "unsupported":
                raise TinyEngineError("prompt exceeds context window") from exc
            raise

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenize(text, parse_special=True))
