from tinyagent.template import render_chat


def test_simple_user_turn():
    out = render_chat([{"role": "user", "content": "Hi"}])
    assert out == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"


def test_system_and_user():
    out = render_chat(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert out == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_no_generation_prompt():
    out = render_chat([{"role": "user", "content": "U"}], add_generation_prompt=False)
    assert out == "<|im_start|>user\nU<|im_end|>\n"


def test_assistant_tool_call_render():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}],
    }
    out = render_chat([msg], add_generation_prompt=False)
    assert out == (
        "<|im_start|>assistant\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        "<|im_end|>\n"
    )


def test_tool_response_render():
    out = render_chat(
        [{"role": "tool", "name": "read_file", "content": "hello"}],
        add_generation_prompt=False,
    )
    assert out == "<|im_start|>user\n<tool_response>\nhello\n</tool_response><|im_end|>\n"


def test_tools_injected_into_system():
    tools = [{"type": "function", "function": {"name": "ping", "description": "p", "parameters": {}}}]
    out = render_chat([{"role": "user", "content": "hi"}], tools=tools)
    assert out.startswith("<|im_start|>system\n")
    assert "<tools>" in out and "</tools>" in out
    assert '"name":"ping"' in out
    assert "<tool_call>" in out  # instructions reference the call format


def test_compact_tool_schema_render_is_short():
    from tinyagent.tools import default_registry

    registry = default_registry(include_network=False)
    compact = render_chat([{"role": "user", "content": "hi"}], tools=registry.schemas(compact=True))
    verbose = render_chat([{"role": "user", "content": "hi"}], tools=registry.schemas(compact=False))

    assert len(compact) < len(verbose)
    assert '"name":"read_file"' in compact
    assert '"args":["path","start_line?","end_line?"]' in compact
    assert "File path relative to the workspace root" not in compact


def test_tool_prefix_render_is_cached():
    from tinyagent.template import _PREFIX_CACHE

    tools = [{"name": "ping", "description": "p", "parameters": {"type": "object"}}]
    _PREFIX_CACHE.clear()

    first = render_chat([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], tools=tools)
    second = render_chat([{"role": "system", "content": "S"}, {"role": "user", "content": "again"}], tools=tools)

    assert first != second
    assert len(_PREFIX_CACHE) == 1


def test_unknown_role_raises():
    import pytest

    with pytest.raises(ValueError):
        render_chat([{"role": "wizard", "content": "x"}])
