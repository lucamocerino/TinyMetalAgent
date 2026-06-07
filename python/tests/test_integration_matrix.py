"""Integration test: agent tools create a Matrix class and unittest suite."""
import importlib.util
import json

from tinyagent.agent import Agent, DEFAULT_SYSTEM_PROMPT
from tinyagent.safety import Approver, Limits
from tinyagent.session import Session
from tinyagent.tools import ToolContext, default_registry


_MATRIX_CLASS = '''\
class Matrix:
    """Small pure-Python dense matrix with matrix multiplication."""

    def __init__(self, rows):
        if not rows:
            raise ValueError("matrix must have at least one row")
        width = len(rows[0])
        if width == 0:
            raise ValueError("matrix must have at least one column")
        normalized = []
        for row in rows:
            if len(row) != width:
                raise ValueError("matrix rows must all have the same length")
            normalized.append(tuple(row))
        self._rows = tuple(normalized)

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]))

    def to_list(self):
        return [list(row) for row in self._rows]

    def __eq__(self, other):
        return isinstance(other, Matrix) and self._rows == other._rows

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        left_rows, left_cols = self.shape
        right_rows, right_cols = other.shape
        if left_cols != right_rows:
            raise ValueError("incompatible matrix dimensions")
        product = []
        for row_index in range(left_rows):
            out_row = []
            for col_index in range(right_cols):
                total = 0
                for inner in range(left_cols):
                    total += self._rows[row_index][inner] * other._rows[inner][col_index]
                out_row.append(total)
            product.append(out_row)
        return Matrix(product)

    def multiply(self, other):
        return self @ other
'''


_MATRIX_TESTS = '''\
import unittest

from matrix import Matrix


class MatrixTests(unittest.TestCase):
    def test_matrix_multiplication_rectangular(self):
        left = Matrix([[1, 2, 3], [4, 5, 6]])
        right = Matrix([[7, 8], [9, 10], [11, 12]])
        self.assertEqual((left @ right).to_list(), [[58, 64], [139, 154]])

    def test_multiply_method_matches_matmul(self):
        left = Matrix([[2, 0], [1, 3]])
        right = Matrix([[4, 5], [6, 7]])
        self.assertEqual(left.multiply(right), left @ right)

    def test_shape_and_copying_output(self):
        matrix = Matrix([[1, 2], [3, 4], [5, 6]])
        copied = matrix.to_list()
        copied[0][0] = 999
        self.assertEqual(matrix.shape, (3, 2))
        self.assertEqual(matrix.to_list()[0][0], 1)

    def test_incompatible_dimensions_raise(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2]]) @ Matrix([[1, 2]])

    def test_ragged_rows_raise(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])


if __name__ == "__main__":
    unittest.main()
'''


class ScriptedEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, max_tokens, on_token=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "done"

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def _tool_call(name, arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def test_agent_tools_create_matrix_class_and_unittests(tmp_path):
    responses = [
        _tool_call("write_file", {"path": "matrix.py", "content": _MATRIX_CLASS}),
        _tool_call("write_file", {"path": "test_matrix.py", "content": _MATRIX_TESTS}),
        _tool_call("run_shell", {"cmd": "python3 -m unittest -q test_matrix && echo MATRIX_TESTS_OK"}),
        _tool_call("finish", {"answer": "Matrix class and multiplication tests pass."}),
    ]
    agent = Agent(
        engine=ScriptedEngine(responses),
        registry=default_registry(include_network=False),
        approver=Approver(mode="yes"),
        tool_context=ToolContext(root=tmp_path, limits=Limits()),
        session=Session(system_prompt=DEFAULT_SYSTEM_PROMPT),
        max_steps=8,
    )

    result = agent.run(
        "Create a pure Python Matrix class and unittest coverage for matrix multiplication."
    )

    assert result.finished
    assert result.answer == "Matrix class and multiplication tests pass."
    assert (tmp_path / "matrix.py").is_file()
    assert (tmp_path / "test_matrix.py").is_file()

    tool_results = [m["content"] for m in agent.session.messages if m["role"] == "tool"]
    shell_outputs = [content for content in tool_results if "exit_code" in content]
    assert shell_outputs
    assert "exit_code: 0" in shell_outputs[-1]
    assert "MATRIX_TESTS_OK" in shell_outputs[-1]

    spec = importlib.util.spec_from_file_location("matrix", tmp_path / "matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result_matrix = module.Matrix([[1, 2, 3]]) @ module.Matrix([[4], [5], [6]])
    assert result_matrix.to_list() == [[32]]
