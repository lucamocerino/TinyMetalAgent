from tinyagent.context import (
    compact_history,
    needs_compaction,
    trim_to_budget,
)
from tinyagent.template import render_chat


def _render(messages):
    return render_chat(messages, add_generation_prompt=False)


def _count(text):
    # ~4 chars per token, good enough for budget tests.
    return max(1, len(text) // 4)


def test_trim_keeps_system_and_recent():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"message number {i} " * 5})
        msgs.append({"role": "assistant", "content": f"reply {i} " * 5})
    trimmed = trim_to_budget(msgs, _render, _count, budget=80, keep_recent=4)
    assert trimmed[0]["role"] == "system"
    # most recent message preserved
    assert trimmed[-1] == msgs[-1]
    assert len(trimmed) < len(msgs)


def test_trim_noop_when_within_budget():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert trim_to_budget(msgs, _render, _count, budget=10_000) == msgs


def test_needs_compaction():
    big = [{"role": "user", "content": "x" * 4000}]
    assert needs_compaction(big, _render, _count, budget=100)
    small = [{"role": "user", "content": "hi"}]
    assert not needs_compaction(small, _render, _count, budget=100)


class _FakeEngine:
    def __init__(self, summary, context_tokens=512):
        self.summary = summary
        self.calls = 0
        self.context_tokens = context_tokens
        self.max_tokens = []

    def generate(self, prompt, max_tokens, on_token=None):
        self.calls += 1
        self.max_tokens.append(max_tokens)
        return self.summary

    def count_tokens(self, text):
        return _count(text)


def test_compact_history_replaces_old_turns():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    engine = _FakeEngine(summary="- did stuff")
    out = compact_history(engine, msgs, keep_recent=4, render_fn=_render)
    assert engine.calls == 1
    assert out[0]["content"] == "SYS"
    assert out[1]["role"] == "system"
    assert "did stuff" in out[1]["content"]
    # last 4 turns preserved verbatim
    assert out[-4:] == msgs[-4:]


def test_compact_history_noop_when_short():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    engine = _FakeEngine(summary="x")
    out = compact_history(engine, msgs, keep_recent=4, render_fn=_render)
    assert out == msgs
    assert engine.calls == 0


def test_compact_history_caps_summary_tokens_to_context():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(8):
        msgs.append({"role": "user", "content": "word " * 8 + str(i)})
    engine = _FakeEngine(summary="- compact", context_tokens=80)

    out = compact_history(engine, msgs, keep_recent=2, max_summary_tokens=256, render_fn=_render)

    assert engine.calls == 1
    assert engine.max_tokens[0] < 256
    assert out[0]["content"] == "SYS"
    assert "compact" in out[1]["content"]
    assert out[-2:] == msgs[-2:]


def test_compact_history_omits_when_prompt_cannot_fit():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(6):
        msgs.append({"role": "user", "content": "word " * 20 + str(i)})
    engine = _FakeEngine(summary="- should not be used", context_tokens=1)

    out = compact_history(engine, msgs, keep_recent=2, max_summary_tokens=256, render_fn=_render)

    assert engine.calls == 0
    assert "too large to summarize" in out[1]["content"]
    assert out[-2:] == msgs[-2:]
