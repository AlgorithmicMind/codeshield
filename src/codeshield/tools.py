# ruff: noqa: UP045
"""Universal agent tool wrapper for CodeShield.

The function returned by ``create_code_execution_tool`` can be registered as a
tool in any agent framework (LangChain, CrewAI, Google Gen AI, etc.). It runs
the provided Python source inside a self-healing sandbox and returns either the
stdout or a structured error report.

Annotations in this module are intentionally *not* postponed: agent SDKs such as
``google-genai`` introspect ``execute_python_code.__annotations__`` at runtime to
build the function-calling schema, and string annotations break that conversion.
"""

from collections.abc import Callable
from typing import Optional

from codeshield.environment import SandboxError
from codeshield.loop import SelfHealingEngine, SelfHealingError
from codeshield.runner import SubprocessRunnerError
from codeshield.schemas import ErrorDiagnosis


def create_code_execution_tool(
    engine: Optional[SelfHealingEngine] = None,
    patch_generator: Optional[Callable[[str, ErrorDiagnosis], Optional[str]]] = None,
) -> Callable[[str], str]:
    """Return a drop-in ``execute_python_code(code: str) -> str`` tool.

    Args:
        engine: Optional ``SelfHealingEngine`` instance. When ``None``, a fresh
            engine is created for each tool call.
        patch_generator: Optional custom patcher ``(code, diagnosis) -> patched``
            forwarded to ``SelfHealingEngine`` when ``engine`` is not provided.

    Returns:
        A callable ready to be registered as an agent tool.
    """

    def execute_python_code(code: str) -> str:
        """Execute Python code in an isolated, self-healing sandbox.

        Use this tool to run numerical, statistical or data-processing
        computations that cannot be done directly in the conversation.

        Args:
            code: A valid Python script as a string.

        Returns:
            The stdout of the script if execution succeeds, or a structured
            error report if it fails after all self-healing attempts.
        """
        _engine = engine or SelfHealingEngine(patch_generator=patch_generator)
        with _engine:
            try:
                result, diagnosis = _engine.run(code)
            except (SelfHealingError, SandboxError, SubprocessRunnerError) as exc:
                return f"error_type: {type(exc).__name__}\nmessage: {exc}"

        if (
            result.exit_code == 0
            and not result.silent_failure_detected
            and not result.timed_out
        ):
            return result.stdout.strip()

        report: list[str] = ["The Python script did not execute successfully."]
        if diagnosis is not None:
            report.append(f"error_type: {diagnosis.error_type}")
            if diagnosis.root_cause_line is not None:
                report.append(f"root_cause_line: {diagnosis.root_cause_line}")
            if diagnosis.message:
                report.append(f"message: {diagnosis.message}")
        if result.stderr.strip():
            report.append(f"stderr: {result.stderr.strip()}")
        if result.timed_out:
            report.append("timed_out: true")
        if result.silent_failure_detected:
            report.append("silent_failure_detected: true")

        return "\n".join(report)

    return execute_python_code
