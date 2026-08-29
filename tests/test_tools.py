"""Unit tests for the CodeShield agent tool wrapper."""

from __future__ import annotations

from codeshield import create_code_execution_tool
from codeshield.loop import SelfHealingEngine


def test_tool_returns_stdout_on_success() -> None:
    """A clean execution returns the stripped stdout."""
    engine = SelfHealingEngine(use_llm=False)
    tool = create_code_execution_tool(engine)

    result = tool("print('hello from tool')")

    assert result == "hello from tool"


def test_tool_reports_controlled_failure() -> None:
    """An unresolved import error surfaces a structured error report."""
    engine = SelfHealingEngine(use_llm=False)
    tool = create_code_execution_tool(engine)

    result = tool("import not_a_real_module_xyz_123")

    assert "error_type" in result
    assert "ModuleNotFoundError" in result
    assert "stderr" in result


def test_tool_without_engine_uses_default() -> None:
    """Calling the tool without an explicit engine still works."""
    tool = create_code_execution_tool()

    result = tool("print(2 + 2)")

    assert result == "4"
