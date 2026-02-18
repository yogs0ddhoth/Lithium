import pytest

from react_agent.tools.calculator import add, divide, multiply


def test_multiply():
    assert multiply.run(tool_input={"a": 1, "b": 4}) == 4
    assert multiply.run(tool_input={"a": 2, "b": 3}) == 6
    assert multiply.run(tool_input={"a": 0, "b": 3}) == 0


def test_add():
    assert add.run(tool_input={"a": 1, "b": 4}) == 5
    assert add.run(tool_input={"a": 2, "b": -3}) == -1
    assert add.run(tool_input={"a": 0, "b": 3}) == 3


def test_divide():
    assert divide.run(tool_input={"a": 1, "b": 4}) == 0.25
    assert divide.run(tool_input={"a": 2, "b": 3}) == (2 / 3)
    # assert divide.run(tool_input={"a": 0, "b": 3}) == 0
    # assert divide.run(tool_input={"a": 0, "b": 3}) == 0
