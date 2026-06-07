"""Chat-call benchmark for the local Qwen model behind tinyagent.

Measures, over a small set of multi-turn chat prompts:
  * prefill latency  — wall time from request to the first decoded token (TTFT)
  * decode throughput — generated tokens per second after the first token
  * end-to-end turn latency

Run:
    python3 -m tinyagent.bench --model ../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Optional

from .engine import TinyEngine
from .template import render_chat
from tinyengine.runtime import detect_arch

_CHAT_TURNS = [
    [{"role": "user", "content": "In one sentence, what is a hash map?"}],
    [
        {"role": "system", "content": "You are a terse coding assistant."},
        {"role": "user", "content": "Write a Python one-liner to reverse a list."},
    ],
    [
        {"role": "user", "content": "What does the Big-O of binary search mean?"},
        {"role": "assistant", "content": "It is O(log n)."},
        {"role": "user", "content": "Why log n and not n?"},
    ],
    [{"role": "user", "content": "Give two reasons to use a dict over a list for lookups."}],
]


class _Meter:
    """Captures time-to-first-token and decode token count via the token callback."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first_token_at: Optional[float] = None
        self.tokens = 0

    def __call__(self, _chunk: str, _token_id: int) -> None:
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.tokens += 1


def _fmt(values: List[float], unit: str) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={statistics.mean(values):.1f}{unit} "
        f"min={min(values):.1f}{unit} max={max(values):.1f}{unit}"
    )


def run_benchmark(model_path: str, max_new_tokens: int, ctx: int) -> int:
    engine = TinyEngine(model_path, context_tokens=ctx)

    ttfts_ms: List[float] = []
    decode_tps: List[float] = []
    turn_ms: List[float] = []

    print(f"model: {model_path}")
    print(f"ctx={ctx} max_new_tokens={max_new_tokens} turns={len(_CHAT_TURNS)}\n")

    for idx, messages in enumerate(_CHAT_TURNS, start=1):
        prompt = render_chat(messages, add_generation_prompt=True)
        prompt_tokens = engine.count_tokens(prompt)
        meter = _Meter()
        t0 = time.perf_counter()
        text = engine.generate(prompt, max_new_tokens, on_token=meter)
        t1 = time.perf_counter()

        ttft_ms = (meter.first_token_at - meter.start) * 1000 if meter.first_token_at else 0.0
        decode_secs = (t1 - meter.first_token_at) if meter.first_token_at else 0.0
        decoded_after_first = max(meter.tokens - 1, 0)
        tps = decoded_after_first / decode_secs if decode_secs > 0 else 0.0
        total_ms = (t1 - t0) * 1000

        ttfts_ms.append(ttft_ms)
        if tps:
            decode_tps.append(tps)
        turn_ms.append(total_ms)

        print(
            f"turn {idx}: prompt_tokens={prompt_tokens:4d} "
            f"gen_tokens={meter.tokens:3d} "
            f"ttft={ttft_ms:7.1f}ms decode={tps:5.1f} tok/s total={total_ms:7.1f}ms"
        )
        preview = text.strip().replace("\n", " ")[:80]
        print(f"         -> {preview!r}")

    print("\nsummary:")
    print(f"  prefill TTFT   : {_fmt(ttfts_ms, 'ms')}")
    print(f"  decode throughput: {_fmt(decode_tps, ' tok/s')}")
    print(f"  turn latency   : {_fmt(turn_ms, 'ms')}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyagent.bench", description=__doc__)
    parser.add_argument(
        "--model",
        default="../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf",
        help="Path to a local Qwen GGUF.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=None, help="Context tokens (default: detected recommendation).")
    args = parser.parse_args(argv)
    ctx = args.ctx if args.ctx is not None else (detect_arch().recommended_max_context or 512)
    return run_benchmark(args.model, args.max_new_tokens, ctx)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
