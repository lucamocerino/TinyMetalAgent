import pytest

from tinyagent.safety import (
    Approver,
    Decision,
    Limits,
    clamp_output,
)


def test_read_and_network_auto_allowed():
    ap = Approver(mode="ask")
    assert ap.decide("read", "read_file", "d").allowed
    assert ap.decide("network", "web_fetch", "d").allowed


def test_write_and_exec_gated_in_ask_mode_without_prompt():
    ap = Approver(mode="ask")  # no prompt_fn -> cannot approve
    assert not ap.decide("write", "write_file", "d").allowed
    assert not ap.decide("exec", "run_shell", "d").allowed


def test_yes_mode_approves_gated():
    ap = Approver(mode="yes")
    assert ap.decide("write", "write_file", "d").allowed
    assert ap.decide("exec", "run_shell", "d").allowed


def test_dry_run_simulates():
    ap = Approver(mode="dry_run")
    decision = ap.decide("exec", "run_shell", "d")
    assert decision.allowed and decision.simulated


def test_no_network_blocks_network():
    ap = Approver(mode="yes", no_network=True)
    assert not ap.decide("network", "web_fetch", "d").allowed


def test_interactive_prompt_answers():
    answers = iter(["n", "y", "a"])
    ap = Approver(mode="ask", prompt_fn=lambda e, n, d: next(answers))
    assert not ap.decide("write", "w", "d").allowed  # n
    assert ap.decide("write", "w", "d").allowed  # y
    assert ap.decide("exec", "x", "d").allowed  # a -> session allow
    # session-allow now auto-approves exec without prompting again
    assert ap.decide("exec", "x", "d").allowed


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Approver(mode="bogus")


def test_clamp_output_short_passthrough():
    assert clamp_output("hello", 100) == "hello"


def test_clamp_output_truncates():
    out = clamp_output("a\n" * 100, 10)
    assert out.startswith("a\na\n")
    assert "truncated" in out


def test_limits_defaults():
    lim = Limits()
    assert lim.shell_timeout == 30.0
    assert lim.max_steps == 12
