import json

from tinyagent.session import Session


def test_session_seeds_system_message():
    s = Session(system_prompt="SYS")
    assert s.messages == [{"role": "system", "content": "SYS"}]


def test_add_turns():
    s = Session(system_prompt="S")
    s.add_user("hi")
    s.add_assistant("hello", tool_calls=[{"name": "f", "arguments": {}}])
    s.add_tool_result("f", "ok")
    roles = [m["role"] for m in s.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert s.messages[2]["tool_calls"][0]["name"] == "f"
    assert s.messages[3] == {"role": "tool", "name": "f", "content": "ok"}


def test_reset():
    s = Session(system_prompt="S")
    s.add_user("hi")
    s.reset()
    assert s.messages == [{"role": "system", "content": "S"}]


def test_save_and_load(tmp_path):
    s = Session(system_prompt="S")
    s.add_user("hi")
    path = tmp_path / "sess.json"
    s.save(str(path))
    data = json.loads(path.read_text())
    assert data["system_prompt"] == "S"
    loaded = Session.load(str(path))
    assert loaded.messages == s.messages
