from pathlib import Path

from tinyagent.agent import Agent, AgentEvent
from tinyagent.safety import Approver, Limits
from tinyagent.session import Session
from tinyagent.tools import ToolContext, default_registry


class ScriptedEngine:
    """Engine stub that returns canned model outputs, one per generate call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.counted = []

    def generate(self, prompt, max_tokens, on_token=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "out of script"

    def count_tokens(self, text):
        self.counted.append(text)
        return max(1, len(text) // 4)


def _make_agent(tmp_path, responses, events=None, max_steps=12):
    engine = ScriptedEngine(responses)
    registry = default_registry(include_network=False)
    approver = Approver(mode="yes")
    ctx = ToolContext(root=tmp_path, limits=Limits())
    session = Session(system_prompt="SYS")
    on_event = events.append if events is not None else None
    return Agent(
        engine=engine,
        registry=registry,
        approver=approver,
        tool_context=ctx,
        session=session,
        max_steps=max_steps,
        on_event=on_event,
    )


def test_write_then_finish(tmp_path):
    responses = [
        '<tool_call>{"name": "write_file", "arguments": {"path": "out.txt", "content": "hi"}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "done"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("create out.txt")
    assert result.finished
    assert result.answer == "done"
    assert result.steps == 2
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_short_history_fit_counts_tokens_once(tmp_path):
    agent = _make_agent(tmp_path, ["ok"])
    agent.session.add_user("hello")

    messages = agent._fit_messages()

    assert messages == agent.session.messages
    assert agent.engine.prompts == []
    assert len(agent.engine.counted) == 1


def test_tool_schemas_cached_for_repeated_turns(tmp_path):
    agent = _make_agent(tmp_path, ["ok"])

    assert agent._tool_schemas() is agent._tool_schemas()


def test_dynamic_tool_window_for_create_ignores_path_words(tmp_path):
    agent = _make_agent(tmp_path, ["ok"])

    tools = agent._select_tool_names("can we create a new tools in a new folder -> test_advance_folder")

    assert "write_file" in tools
    assert "finish" in tools
    assert "run_shell" in tools


def test_dynamic_tool_window_for_unit_tests(tmp_path):
    agent = _make_agent(tmp_path, ["ok"])

    tools = agent._select_tool_names("can you add some unittests ?")

    assert "glob" in tools
    assert "read_file" in tools
    assert "write_file" in tools
    assert "run_shell" in tools
    assert "finish" in tools


def test_dynamic_act_prompt_fits_small_budget_after_chat(tmp_path):
    agent = _make_agent(tmp_path, ["ok"])
    agent.session.add_user("hey")
    agent.session.add_assistant("Hello! How can I assist you today?")
    agent.session.add_user("can we create a new tools in a new folder -> test_advance_folder")
    tools = agent._select_tool_names(agent.session.messages[-1]["content"])

    messages = agent._fit_messages(tool_names=tools)
    prompt = agent._render(messages, tool_names=tools)

    assert "write_file" in prompt
    assert "grep" not in prompt
    assert agent.engine.count_tokens(prompt) <= agent.token_budget


def test_no_tool_call_is_final_answer(tmp_path):
    agent = _make_agent(tmp_path, ["The answer is 42."])
    result = agent.run("what is the answer")
    assert result.finished
    assert result.answer == "The answer is 42."
    assert result.steps == 1
    assert "<tools>" not in agent.engine.prompts[0]


def test_greeting_routes_to_chat_without_tools(tmp_path):
    agent = _make_agent(tmp_path, ["Hello!"])
    result = agent.run("hey")
    assert result.finished
    assert result.answer == "Hello!"
    assert "<tools>" not in agent.engine.prompts[0]


def test_markdown_code_dump_is_nudged_then_writes(tmp_path):
    # Stronger models sometimes answer "create a file" with a markdown tutorial
    # instead of a tool call; the agent must nudge it back to the tool protocol.
    responses = [
        "Here is the file:\n```python\ndef f():\n    return 1\n```\n",
        '<tool_call>{"name": "write_file", "arguments": {"path": "f.py", "content": "def f():\\n    return 1\\n"}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "done"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("create f.py")
    assert result.finished
    assert (tmp_path / "f.py").read_text() == "def f():\n    return 1\n"
    assert result.steps == 3
    # A reminder (user turn) was injected after the prose response.
    user_turns = [m for m in agent.session.messages if m["role"] == "user"]
    assert any("did not call a tool" in m["content"] for m in user_turns)


def test_action_prose_is_nudged_then_writes(tmp_path):
    responses = [
        "Sure, I can create that folder.",
        '<tool_call>{"name": "write_file", "arguments": {"path": "test_advance_folder/tool.txt", "content": "ok"}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "done"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("create a new tool in test_advance_folder")
    assert result.finished
    assert (tmp_path / "test_advance_folder" / "tool.txt").read_text() == "ok"
    user_turns = [m for m in agent.session.messages if m["role"] == "user"]
    assert any("did not call a tool" in m["content"] for m in user_turns)
    assert "create a new tool in test_advance_folder" in agent.engine.prompts[1]


def test_action_repeated_prose_does_not_finish_successfully(tmp_path):
    responses = [
        "I cannot write tests directly.",
        "Please provide code and I can suggest tests.",
    ]
    agent = _make_agent(tmp_path, responses)

    result = agent.run("can you add some unittests ?")

    assert not result.finished
    assert result.answer == "I could not produce a valid tool call for this task."
    assert "<tools>" in agent.engine.prompts[0]
    assert "can you add some unittests ?" in agent.engine.prompts[1]


def test_malformed_tool_call_is_nudged(tmp_path):
    responses = [
        "<tool_call>{this is not valid json}</tool_call>",
        '<tool_call>{"name": "finish", "arguments": {"answer": "ok"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("go")
    assert result.finished
    assert result.steps == 2
    user_turns = [m for m in agent.session.messages if m["role"] == "user"]
    assert any("malformed" in m["content"] for m in user_turns)


def test_fenced_answer_after_tool_ran_is_not_nudged(tmp_path):
    # Once real work has happened, a fenced explanatory answer is a legitimate
    # final response and must not trigger a retry.
    responses = [
        '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>',
        "Done. The function is:\n```python\ndef f():\n    return 1\n```\n",
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("inspect then summarise")
    assert result.finished
    assert result.steps == 2
    assert "```python" in result.answer


def test_question_with_code_block_is_not_nudged(tmp_path):
    # A conceptual question answered with an illustrative code block is a valid,
    # informative reply — it must terminate the loop, not be forced into a write.
    fenced = "A linked list chains nodes:\n```python\nclass Node: ...\n```\n"
    agent = _make_agent(tmp_path, [fenced])
    result = agent.run("explain a linked list")
    assert result.finished
    assert result.steps == 1
    assert result.answer == fenced.strip() or result.answer == fenced
    # No nudge user-turn was injected.
    user_turns = [m for m in agent.session.messages if m["role"] == "user"]
    assert not any("did not call a tool" in m["content"] for m in user_turns)
    assert "<tools>" not in agent.engine.prompts[0]


def test_chat_mode_answers_without_tools(tmp_path):
    # chat() returns a single-turn prose answer and never runs tools, even if the
    # model emits stray tool-call markup.
    agent = _make_agent(
        tmp_path,
        ["A stack is LIFO.\n<tool_call>{\"name\": \"write_file\"}</tool_call>"],
    )
    result = agent.chat("what is a stack")
    assert result.finished
    assert result.steps == 1
    assert result.answer == "A stack is LIFO."
    assert not list(tmp_path.iterdir())  # nothing written
    # Chat-only budgeting and generation carry no tool schemas.
    assert "<tools>" not in agent.engine.counted[0]
    assert "<tools>" not in agent.engine.prompts[0]


class StreamingEngine:
    """Engine stub that streams a canned response token-by-token via on_token."""

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, max_tokens, on_token=None):
        for i, ch in enumerate(self.text):
            if on_token is not None:
                on_token(ch, i)
        return self.text

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def test_chat_streams_tokens_via_on_token(tmp_path):
    engine = StreamingEngine("hello there")
    registry = default_registry(include_network=False)
    agent = Agent(
        engine=engine,
        registry=registry,
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session(system_prompt="SYS"),
    )
    received = []
    result = agent.chat("hi", on_token=lambda chunk, _i: received.append(chunk))
    # on_token fired once per character (incremental streaming, not one dump).
    assert "".join(received) == "hello there"
    assert len(received) == len("hello there")
    assert result.answer == "hello there"


def test_unknown_tool_reported_then_finish(tmp_path):
    responses = [
        '<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "ok"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses)
    result = agent.run("go")
    assert result.finished
    tool_results = [m for m in agent.session.messages if m["role"] == "tool"]
    assert "unknown tool" in tool_results[0]["content"]


def test_events_emitted(tmp_path):
    events = []
    responses = [
        '<tool_call>{"name": "write_file", "arguments": {"path": "x.txt", "content": "y"}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "fin"}}</tool_call>',
    ]
    agent = _make_agent(tmp_path, responses, events=events)
    agent.run("go")
    kinds = [e.kind for e in events]
    assert "status" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert any(e.kind == "status" and "model response" in e.text for e in events)
    assert any(e.kind == "tool_result" and e.elapsed_ms >= 0 for e in events)
    assert kinds[-1] == "final"


def test_dry_run_does_not_write(tmp_path):
    engine = ScriptedEngine([
        '<tool_call>{"name": "write_file", "arguments": {"path": "z.txt", "content": "y"}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"answer": "k"}}</tool_call>',
    ])
    agent = Agent(
        engine=engine,
        registry=default_registry(include_network=False),
        approver=Approver(mode="dry_run"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session(system_prompt="S"),
    )
    agent.run("go")
    assert not (tmp_path / "z.txt").exists()


def test_max_steps_unfinished(tmp_path):
    # Always emits a read tool call, never finishes.
    loop_call = '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>'
    agent = _make_agent(tmp_path, [loop_call] * 5, max_steps=3)
    result = agent.run("loop forever")
    assert not result.finished
    assert result.steps == 3
