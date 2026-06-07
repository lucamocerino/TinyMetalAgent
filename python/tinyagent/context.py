"""Context budgeting and conversation compaction.

The v1 engine re-prefills the whole transcript every turn (no KV reuse), so keeping
the rendered prompt within a token budget is essential. Two mechanisms:

* ``trim_to_budget`` — cheap structural trimming: always keep the leading system
  message and the most recent turns; drop the oldest middle turns until it fits.
* ``compact_history`` — the ``/compact`` behaviour: ask the local model to summarize
  older turns into a single note, preserving the system message and recent turns.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

Message = Dict[str, Any]
CountFn = Callable[[str], int]


def _split_system(messages: Sequence[Message]):
    if messages and messages[0].get("role") == "system":
        return [messages[0]], list(messages[1:])
    return [], list(messages)


def trim_to_budget(
    messages: Sequence[Message],
    render_fn: Callable[[Sequence[Message]], str],
    count_fn: CountFn,
    budget: int,
    keep_recent: int = 6,
    min_recent: int = 2,
) -> List[Message]:
    """Drop oldest non-system turns until the rendered prompt fits ``budget``.

    Prefers to retain the most recent ``keep_recent`` turns, but will keep dropping
    down to ``min_recent`` turns when that is the only way to fit the budget — the
    engine hard-fails if the prompt plus the reply exceed the context window, so an
    over-budget prompt is never safe to return.
    """
    system, body = _split_system(messages)
    if count_fn(render_fn(list(system) + body)) <= budget:
        return list(system) + body

    floor = keep_recent if keep_recent >= min_recent else min_recent
    while len(body) > floor:
        body = body[1:]
        if count_fn(render_fn(list(system) + body)) <= budget:
            return list(system) + body
    while len(body) > min_recent:
        body = body[1:]
        if count_fn(render_fn(list(system) + body)) <= budget:
            break
    return list(system) + body


def needs_compaction(
    messages: Sequence[Message],
    render_fn: Callable[[Sequence[Message]], str],
    count_fn: CountFn,
    budget: int,
) -> bool:
    return count_fn(render_fn(list(messages))) > budget


def compact_history(
    engine,
    messages: Sequence[Message],
    keep_recent: int = 4,
    max_summary_tokens: int = 256,
    render_fn: Optional[Callable[[Sequence[Message]], str]] = None,
) -> List[Message]:
    """Summarize older turns into one system note via the local model.

    Keeps the original system message and the last ``keep_recent`` turns verbatim;
    everything in between is replaced by a single ``system`` summary message.
    """
    from .template import render_chat

    render = render_fn or (lambda m: render_chat(m, add_generation_prompt=False))
    system, body = _split_system(messages)
    if len(body) <= keep_recent:
        return list(messages)

    older = body[:-keep_recent]
    recent = body[-keep_recent:]
    context_tokens = int(getattr(engine, "context_tokens", 512) or 512)

    def build_prompt(items: Sequence[Message]) -> str:
        transcript = render(items).replace("<|im_start|>", "").replace("<|im_end|>", "")
        return (
            "<|im_start|>user\nSummarize the following coding-session transcript into a "
            "compact set of bullet points capturing decisions, files touched, and open "
            "tasks. Be concise.\n\n"
            + transcript
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    summary_prompt = build_prompt(older)
    while older and engine.count_tokens(summary_prompt) >= context_tokens:
        older = older[1:]
        summary_prompt = build_prompt(older)

    prompt_tokens = engine.count_tokens(summary_prompt)
    room = context_tokens - prompt_tokens
    if room < 1:
        summary = "Earlier conversation was too large to summarize within the available context."
    else:
        summary_tokens = min(max_summary_tokens, room)
        summary = engine.generate(summary_prompt, summary_tokens).strip()
    note = {
        "role": "system",
        "content": "Summary of earlier conversation:\n" + summary,
    }
    return list(system) + [note] + recent
