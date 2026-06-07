"""Integration tests for the Two Sum acceptance task.

Two layers:

* ``test_two_sum_pipeline_*`` — deterministic. A scripted engine drives the REAL
  file/shell tools to write a Two Sum solution and verify it against the three
  LeetCode example cases via ``run_shell``. This proves the agent loop + tools +
  approval + verification harness work end to end, independent of model quality.

* ``test_two_sum_real_model`` — opt-in (set ``TINYAGENT_RUN_MODEL=1``). Runs the
  real local Qwen Coder model through the agent and asserts the *infrastructure*
  drives tools and finishes. Model-output correctness is not asserted here.
"""
import os
from pathlib import Path

import pytest

from tinyagent.agent import Agent, DEFAULT_SYSTEM_PROMPT
from tinyagent.safety import Approver, Limits
from tinyagent.session import Session
from tinyagent.tools import ToolContext, default_registry

_CORRECT_SOLUTION = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, n in enumerate(nums):\n"
    "        if target - n in seen:\n"
    "            return [seen[target - n], i]\n"
    "        seen[n] = i\n"
    "    return []\n"
)

_TEST_HARNESS = (
    "from two_sum import two_sum\n"
    "assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]\n"
    "assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]\n"
    "assert sorted(two_sum([3, 3], 6)) == [0, 1]\n"
    "print('ALL PASS')\n"
)


class ScriptedEngine:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, max_tokens, on_token=None):
        return self.responses.pop(0) if self.responses else "done"

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def _tool_call(name, arguments):
    import json

    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'


def test_two_sum_pipeline_writes_and_verifies(tmp_path):
    """Full agent pipeline: write solution + harness, run it, see all cases pass."""
    responses = [
        _tool_call("write_file", {"path": "two_sum.py", "content": _CORRECT_SOLUTION}),
        _tool_call("write_file", {"path": "check.py", "content": _TEST_HARNESS}),
        _tool_call("run_shell", {"cmd": "python3 check.py"}),
        _tool_call("finish", {"answer": "Two Sum solved; all cases pass."}),
    ]
    agent = Agent(
        engine=ScriptedEngine(responses),
        registry=default_registry(include_network=False),
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session(system_prompt=DEFAULT_SYSTEM_PROMPT),
        max_steps=8,
    )
    result = agent.run("Solve the Two Sum problem and verify it.")

    assert result.finished
    assert (tmp_path / "two_sum.py").exists()
    # The run_shell tool result for the verification step must report success.
    tool_results = [m["content"] for m in agent.session.messages if m["role"] == "tool"]
    shell_outputs = [c for c in tool_results if "exit_code" in c]
    assert shell_outputs, "expected a run_shell result"
    assert "ALL PASS" in shell_outputs[-1]
    assert "exit_code: 0" in shell_outputs[-1]


def test_two_sum_solution_is_correct_directly(tmp_path):
    """The reference solution itself satisfies the three LeetCode examples."""
    (tmp_path / "two_sum.py").write_text(_CORRECT_SOLUTION)
    import importlib.util

    spec = importlib.util.spec_from_file_location("two_sum", tmp_path / "two_sum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sorted(module.two_sum([2, 7, 11, 15], 9)) == [0, 1]
    assert sorted(module.two_sum([3, 2, 4], 6)) == [1, 2]
    assert sorted(module.two_sum([3, 3], 6)) == [0, 1]


_MODEL_PATH_CANDIDATES = [
    os.environ.get("TINYAGENT_MODEL", ""),
    "../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf",
    "models/qwen2.5-coder-3b-instruct-q4_0-te.gguf",
]


def _find_model():
    for cand in _MODEL_PATH_CANDIDATES:
        if cand and Path(cand).is_file():
            return cand
    return None


@pytest.mark.skipif(
    os.environ.get("TINYAGENT_RUN_MODEL") != "1",
    reason="set TINYAGENT_RUN_MODEL=1 to run the real local-model integration test",
)
def test_two_sum_real_model(tmp_path):
    model = _find_model()
    if model is None:
        pytest.skip("no local Qwen GGUF found")
    from tinyagent.engine import TinyEngine

    agent = Agent(
        engine=TinyEngine(model, context_tokens=2048),
        registry=default_registry(include_network=False),
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session(system_prompt=DEFAULT_SYSTEM_PROMPT),
        max_steps=4,
        max_new_tokens=320,
    )
    task = (
        "Write a file named two_sum.py defining a function two_sum(nums, target) "
        "that returns the two indices whose values sum to target. Use the write_file "
        "tool, then call finish."
    )
    result = agent.run(task)
    # Infrastructure assertions only: the loop ran tools and terminated.
    assert result.steps >= 1
    assert result.finished
    assert any(m["role"] == "tool" for m in agent.session.messages)
