from tinyagent.toolcall import parse_tool_calls


def test_canonical_tool_call():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    res = parse_tool_calls(text)
    assert res.has_calls
    assert res.tool_calls[0].name == "read_file"
    assert res.tool_calls[0].arguments == {"path": "a.py"}
    assert res.content == ""


def test_tool_call_with_leading_text():
    text = 'Let me read it.\n<tool_call>{"name": "list_dir", "arguments": {}}</tool_call>'
    res = parse_tool_calls(text)
    assert res.tool_calls[0].name == "list_dir"
    assert res.content == "Let me read it."


def test_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    res = parse_tool_calls(text)
    assert [c.name for c in res.tool_calls] == ["a", "b"]
    assert res.tool_calls[1].arguments == {"x": 1}


def test_bare_json_object():
    text = '{"name": "write_file", "arguments": {"path": "x", "content": "y"}}'
    res = parse_tool_calls(text)
    assert res.has_calls
    assert res.tool_calls[0].name == "write_file"
    assert res.tool_calls[0].arguments["content"] == "y"


def test_line_based_fallback():
    text = 'TOOL: run_shell {"cmd": "ls"}'
    res = parse_tool_calls(text)
    assert res.tool_calls[0].name == "run_shell"
    assert res.tool_calls[0].arguments == {"cmd": "ls"}


def test_parameters_alias_and_trailing_comma_repair():
    text = '<tool_call>{"name": "f", "parameters": {"a": 1,}}</tool_call>'
    res = parse_tool_calls(text)
    assert res.tool_calls[0].arguments == {"a": 1}


def test_code_fenced_bare_json():
    text = '```json\n{"name": "f", "arguments": {"a": 2}}\n```'
    res = parse_tool_calls(text)
    assert res.has_calls
    assert res.tool_calls[0].arguments == {"a": 2}


def test_plain_text_no_calls():
    res = parse_tool_calls("The answer is 42.")
    assert not res.has_calls
    assert res.content == "The answer is 42."


def test_malformed_never_raises():
    text = '<tool_call>{not json at all}</tool_call>'
    res = parse_tool_calls(text)
    assert not res.has_calls
    assert res.malformed


def test_multiple_bare_json_objects():
    text = (
        '{"name": "write_file", "arguments": {"path": "a.py", "content": "x\\ny"}}\n'
        '{"name": "finish", "arguments": {"answer": "done"}}'
    )
    res = parse_tool_calls(text)
    assert [c.name for c in res.tool_calls] == ["write_file", "finish"]
    assert res.tool_calls[0].arguments["content"] == "x\ny"


def test_bare_json_with_braces_in_string():
    text = '{"name": "write_file", "arguments": {"content": "if (x) { y(); }"}}'
    res = parse_tool_calls(text)
    assert res.tool_calls[0].arguments["content"] == "if (x) { y(); }"


def test_bare_non_tool_json_is_not_a_call():
    res = parse_tool_calls('{"foo": 1, "bar": 2}')
    assert not res.has_calls


def test_triple_quoted_content_is_repaired():
    # The small model often writes multi-line content as a Python triple-quoted
    # string, which is invalid JSON. The parser should recover it.
    text = (
        '```json\n{"name": "write_file", "arguments": {"path": "lru.py", "content": """'
        "class LRUCache:\n    def __init__(self, capacity):\n        self.cap = {capacity}\n"
        '"""}}\n```'
    )
    res = parse_tool_calls(text)
    assert res.has_calls
    assert res.tool_calls[0].name == "write_file"
    assert res.tool_calls[0].arguments["path"] == "lru.py"
    assert res.tool_calls[0].arguments["content"].startswith("class LRUCache:")
    assert "self.cap = {capacity}" in res.tool_calls[0].arguments["content"]


def test_triple_quoted_inside_tool_call_tags():
    text = '<tool_call>{"name": "write_file", "arguments": {"path": "a.py", "content": """print(1)"""}}</tool_call>'
    res = parse_tool_calls(text)
    assert res.has_calls
    assert res.tool_calls[0].arguments["content"] == "print(1)"
