"""Functional unit tests for syntax and runtime error capture."""

from __future__ import annotations

import pytest

from codeshield.analyzer import validate_syntax_and_safety
from codeshield.classifier import TracebackClassifier
from codeshield.environment import SandboxManager
from codeshield.loop import SelfHealingEngine
from codeshield.runner import SubprocessRunner
from codeshield.schemas import (
    ExecutionResult,
)


@pytest.fixture
def sandbox() -> SandboxManager:
    """Provide a reusable sandbox manager."""
    return SandboxManager()


@pytest.fixture
def runner(sandbox: SandboxManager) -> SubprocessRunner:
    """Provide a subprocess runner backed by the sandbox fixture."""
    return SubprocessRunner(sandbox)


def test_syntax_error_is_captured() -> None:
    """A script with a syntax error is flagged by the AST validator."""
    code = 'print("hello"'
    report = validate_syntax_and_safety(code)

    assert not report.is_valid
    assert report.violations
    assert "SyntaxError" in report.violations[0]
    assert report.exception is not None


def test_generic_except_is_flagged() -> None:
    """Overly broad exception handlers are reported as safety violations."""
    code = """
try:
    x = 1
except:
    pass
"""
    report = validate_syntax_and_safety(code)
    assert not report.is_valid
    assert any("Bare ``except:" in v for v in report.violations)


def test_dangerous_eval_is_flagged() -> None:
    """Direct calls to ``eval`` are detected by the safety visitor."""
    code = "result = eval('1 + 1')"
    report = validate_syntax_and_safety(code)

    assert not report.is_valid
    assert any("eval" in v for v in report.violations)


def test_traceback_classifier_extracts_name_error() -> None:
    """A traceback is parsed into a structured ``ErrorDiagnosis``."""
    code = "\n".join([
        "import math",
        "value = undefined_variable",
        "print(value)",
    ])
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "script.py", line 2, in <module>\n'
        "    value = undefined_variable\n"
        "NameError: name 'undefined_variable' is not defined\n"
    )
    classifier = TracebackClassifier()
    diagnosis = classifier.classify(code, stderr)

    assert diagnosis is not None
    assert diagnosis.error_type == "NameError"
    assert diagnosis.root_cause_line == 2
    assert "undefined_variable" in diagnosis.message
    assert any("value = undefined_variable" in ctx for ctx in diagnosis.context)


def test_traceback_classifier_returns_none_for_clean_output() -> None:
    """No diagnosis is produced when ``stderr`` does not contain a traceback."""
    code = "print('ok')"
    stderr = ""
    classifier = TracebackClassifier()

    assert classifier.classify(code, stderr) is None


def test_runtime_zero_division_is_captured_and_classified(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """A runtime error is executed, captured and classified correctly."""
    sandbox.create()
    try:
        code = "print(1 / 0)"
        result = runner.run(code)

        assert isinstance(result, ExecutionResult)
        assert result.exit_code != 0
        assert "Traceback" in result.stderr
        assert "ZeroDivisionError" in result.stderr

        classifier = TracebackClassifier()
        diagnosis = classifier.classify(code, result.stderr)

        assert diagnosis is not None
        assert diagnosis.error_type == "ZeroDivisionError"
        assert diagnosis.root_cause_line == 1
    finally:
        sandbox.cleanup()


def test_silent_failure_pattern_is_detected(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """A zero exit code with a suspicious pattern is flagged as silent failure."""
    sandbox.create()
    try:
        code = "print('Pipeline failed')"
        result = runner.run(code)

        assert result.exit_code == 0
        assert result.silent_failure_detected
    finally:
        sandbox.cleanup()


def test_self_healing_engine_runs_clean_code() -> None:
    """The self-healing engine returns a clean result for valid code."""
    engine = SelfHealingEngine()
    with engine:
        result, diagnosis = engine.run("print('hello world')")

        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
        assert diagnosis is None


def test_runner_enforces_timeout(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """A long-running process is terminated when it exceeds its timeout."""
    sandbox.create()
    try:
        code = "import time; time.sleep(60)"
        result = runner.run(code, timeout=1.0)
        assert result.timed_out
        assert result.exit_code != 0
    finally:
        sandbox.cleanup()


def test_sandbox_venv_backend() -> None:
    """The venv backend creates an isolated environment without uv."""
    sandbox = SandboxManager(backend="venv")
    with sandbox:
        assert sandbox.python_executable.exists()


def test_nested_exec_is_flagged() -> None:
    """Nested eval/exec calls are still detected by the safety visitor."""
    code = 'exec("eval(\'1 + 1\')")'
    report = validate_syntax_and_safety(code)
    assert not report.is_valid
    assert any("eval" in v or "exec" in v for v in report.violations)


def test_unrecoverable_syntax_error() -> None:
    """Severely malformed code is caught as a syntax error."""
    code = "def foo("
    report = validate_syntax_and_safety(code)
    assert not report.is_valid
    assert report.exception is not None
    assert "SyntaxError" in report.violations[0]


def test_dangerous_imported_call_is_flagged() -> None:
    """Calls to dangerous functions imported from modules are detected."""
    code = "import os; os.system('echo pwned')"
    report = validate_syntax_and_safety(code)
    assert not report.is_valid
    assert any(
        "os.system" in v or "system" in v or "dangerous" in v.lower()
        for v in report.violations
    )


def test_silent_failure_dataframe_empty(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """A zero exit with an empty DataFrame warning is flagged."""
    sandbox.create()
    try:
        code = "print('empty DataFrame')"
        result = runner.run(code)
        assert result.exit_code == 0
        assert result.silent_failure_detected
    finally:
        sandbox.cleanup()


def test_silent_failure_nan_pattern(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """A zero exit with NaN warnings is flagged."""
    sandbox.create()
    try:
        code = "print('all NaN')"
        result = runner.run(code)
        assert result.exit_code == 0
        assert result.silent_failure_detected
    finally:
        sandbox.cleanup()


def test_self_healing_engine_fixes_name_error() -> None:
    """The local fallback heals a missing import NameError."""
    engine = SelfHealingEngine(use_llm=False)
    with engine:
        result, diagnosis = engine.run("print(math.sqrt(16))")

        assert result.exit_code == 0
        assert result.stdout.strip() == "4.0"
        assert diagnosis is None
