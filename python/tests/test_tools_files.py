import pytest

from tinyagent.safety import Limits
from tinyagent.tools import ToolContext
from tinyagent.tools.files import (
    ToolError,
    apply_patch,
    apply_unified_diff,
    glob_tool,
    grep_tool,
    list_dir,
    read_file,
    resolve_path,
    write_file,
)
from tinyagent.tools.shell import run_shell


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("hello\nworld\n", encoding="utf-8")
    return ToolContext(root=tmp_path, limits=Limits())


def test_read_file_numbered(ctx):
    out = read_file({"path": "a.py"}, ctx)
    assert "1\tdef f():" in out
    assert "2\t    return 1" in out


def test_read_file_range(ctx):
    out = read_file({"path": "a.py", "start_line": 2, "end_line": 2}, ctx)
    assert out == "2\t    return 1"


def test_read_missing_raises(ctx):
    with pytest.raises(ToolError):
        read_file({"path": "nope.py"}, ctx)


def test_list_dir(ctx):
    out = list_dir({"path": "."}, ctx)
    assert "sub/" in out
    assert "a.py" in out


def test_glob(ctx):
    out = glob_tool({"pattern": "**/*.py"}, ctx)
    assert out == "a.py"


def test_grep(ctx):
    out = grep_tool({"pattern": "world", "path": "sub"}, ctx)
    assert "b.txt:2:world" in out


def test_write_file(ctx):
    msg = write_file({"path": "new/c.txt", "content": "hi"}, ctx)
    assert "wrote" in msg
    assert (ctx.root / "new" / "c.txt").read_text() == "hi"


def test_resolve_path_escape_blocked(ctx):
    with pytest.raises(ToolError):
        resolve_path(ctx, "../../etc/passwd")


def test_apply_unified_diff_basic():
    original = "line1\nline2\nline3\n"
    diff = (
        "--- a\n+++ b\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+line2-changed\n"
        " line3\n"
    )
    updated = apply_unified_diff(original, diff)
    assert updated == "line1\nline2-changed\nline3\n"


def test_apply_unified_diff_add_lines():
    original = "a\nb\n"
    diff = "@@ -1,2 +1,3 @@\n a\n+inserted\n b\n"
    updated = apply_unified_diff(original, diff)
    assert updated == "a\ninserted\nb\n"


def test_apply_unified_diff_context_mismatch():
    with pytest.raises(ToolError):
        apply_unified_diff("a\nb\n", "@@ -1,2 +1,2 @@\n X\n-b\n+c\n")


def test_apply_unified_diff_no_hunk():
    with pytest.raises(ToolError):
        apply_unified_diff("a\n", "not a diff")


def test_apply_patch_roundtrip(ctx):
    diff = "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    apply_patch({"path": "a.py", "diff": diff}, ctx)
    assert (ctx.root / "a.py").read_text() == "def f():\n    return 2\n"


def test_run_shell_captures_output(ctx):
    out = run_shell({"cmd": "echo hello"}, ctx)
    assert "exit_code: 0" in out
    assert "hello" in out


def test_run_shell_nonzero_exit(ctx):
    out = run_shell({"cmd": "exit 3"}, ctx)
    assert "exit_code: 3" in out


def test_run_shell_timeout(ctx):
    with pytest.raises(ToolError):
        run_shell({"cmd": "sleep 5", "timeout": 0.2}, ctx)
