from argparse import Namespace

from tinyagent import cli
from tinyagent.agent import Agent
from tinyagent.agent import AgentEvent
from tinyagent.safety import Approver, Limits
from tinyagent.session import Session
from tinyagent.tools import ToolContext, default_registry


class FakeArch:
    recommended_max_context = 512


class FakeEngine:
    def __init__(self, model_path, context_tokens):
        self.model_path = model_path
        self.context_tokens = context_tokens

    def generate(self, prompt, max_tokens, on_token=None):
        return "ok"

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def _args(**overrides):
    values = dict(
        model=None,
        ctx=None,
        daemon=False,
        no_network=True,
        yes=True,
        dry_run=False,
        root=".",
        shell_timeout=30.0,
        max_steps=12,
        max_new_tokens=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_build_agent_uses_detected_context_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_find_model", lambda _explicit: "model.gguf")
    monkeypatch.setattr(cli, "detect_arch", lambda: FakeArch())
    monkeypatch.setattr(cli, "TinyEngine", FakeEngine)

    agent = cli.build_agent(_args(), Session("SYS"), None)

    assert agent.engine.context_tokens == 512
    assert agent.max_new_tokens == 128
    assert agent.token_budget == 320


def test_build_agent_respects_explicit_context_and_tokens(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_find_model", lambda _explicit: "model.gguf")
    monkeypatch.setattr(cli, "TinyEngine", FakeEngine)

    agent = cli.build_agent(_args(ctx=1024, max_new_tokens=64), Session("SYS"), None)

    assert agent.engine.context_tokens == 1024
    assert agent.max_new_tokens == 64
    assert agent.token_budget == 896


def test_slash_matches_complete_commands():
    assert "/ask" in cli._slash_matches("/a")
    assert "/context" in cli._slash_matches("/c")
    assert "/diff" in cli._slash_matches("/d")
    assert "/approve auto" in cli._slash_matches("/approve")
    assert cli._slash_matches("ask") == []
    assert cli._slash_matches("/ask hello") == []


def test_mention_matches_complete_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")

    assert "@src/" in cli._mention_matches("@s", tmp_path)
    assert "@src/main.py" in cli._mention_matches("@src/m", tmp_path)


def test_expand_mentions_attaches_file(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    ctx = ToolContext(root=tmp_path, limits=Limits())

    expanded = cli._expand_mentions("explain @main.py", ctx)

    assert "Attached context:" in expanded
    assert "--- @main.py ---" in expanded
    assert "print('hi')" in expanded


def test_expand_mentions_blocks_escape(tmp_path):
    ctx = ToolContext(root=tmp_path, limits=Limits())

    expanded = cli._expand_mentions("read @../secret.txt", ctx)

    assert "mention error" in expanded


def test_shell_escape_runs_in_workspace(tmp_path, capsys):
    renderer = cli.Renderer(verbose=False)
    agent = Agent(
        engine=FakeEngine("model.gguf", 512),
        registry=default_registry(include_network=False),
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session("SYS"),
    )

    assert cli._run_shell_escape("printf hello", agent, renderer) == 0
    assert "hello" in capsys.readouterr().out


def test_context_report_shows_token_usage(tmp_path):
    agent = Agent(
        engine=FakeEngine("model.gguf", 512),
        registry=default_registry(include_network=False),
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session("SYS"),
        max_new_tokens=32,
    )
    agent.session.add_user("create file")

    report = cli._context_report(agent)

    assert "messages:" in report
    assert "chat prompt:" in report
    assert "ACT prompt:" in report
    assert "ACT tools:" in report


def test_renderer_hides_status_details_by_default(capsys):
    renderer = cli.Renderer(verbose=False)

    renderer(AgentEvent(kind="status", text="rendered ACT prompt", elapsed_ms=12))
    renderer(AgentEvent(kind="step", step=1))
    renderer(AgentEvent(kind="tool_call", name="write_file", arguments={"path": "x"}))
    renderer(AgentEvent(kind="tool_result", name="write_file", text="wrote lots", elapsed_ms=3))

    out = capsys.readouterr()
    assert "rendered ACT prompt" not in out.err
    assert "step" not in out.err
    assert "path='x'" not in out.out
    assert "wrote lots" not in out.out
    assert "→ tool write_file" in out.out
    assert "✓ write_file" in out.out


def test_renderer_verbose_shows_status_and_tool_details(capsys):
    renderer = cli.Renderer(verbose=True)

    renderer(AgentEvent(kind="status", text="rendered ACT prompt", elapsed_ms=12))
    renderer(AgentEvent(kind="step", step=1))
    renderer(AgentEvent(kind="tool_call", name="write_file", arguments={"path": "x"}))
    renderer(AgentEvent(kind="tool_result", name="write_file", text="wrote lots", elapsed_ms=3))

    out = capsys.readouterr()
    combined = out.out + out.err
    assert "rendered ACT prompt" in combined
    assert "— step 1 —" in combined
    assert "path='x'" in combined
    assert "wrote lots" in combined
