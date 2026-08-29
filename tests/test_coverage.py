"""Additional functional tests to raise code coverage above 80%."""

from __future__ import annotations

import runpy
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from codeshield.classifier import TracebackClassifier
from codeshield.environment import SandboxError, SandboxManager
from codeshield.loop import (
    GeminiPatchGenerator,
    SelfHealingEngine,
    SelfHealingError,
)
from codeshield.runner import SubprocessRunner
from codeshield.schemas import CodeExecutionRequest, ErrorDiagnosis


@pytest.fixture
def sandbox() -> SandboxManager:
    """Provide a reusable sandbox manager."""
    return SandboxManager()


@pytest.fixture
def runner(sandbox: SandboxManager) -> SubprocessRunner:
    """Provide a subprocess runner backed by the sandbox fixture."""
    return SubprocessRunner(sandbox)


def test_classifier_extract_missing_name() -> None:
    """The classifier extracts the missing name from a NameError."""
    classifier = TracebackClassifier()
    diagnosis = ErrorDiagnosis(
        error_type="NameError",
        root_cause_line=2,
        message="name 'undefined_value' is not defined",
        context=["2: print(undefined_value)"],
    )

    assert classifier.extract_missing_name(diagnosis) == "undefined_value"


def test_classifier_extract_missing_module() -> None:
    """The classifier extracts the missing module from an ImportError."""
    classifier = TracebackClassifier()
    diagnosis = ErrorDiagnosis(
        error_type="ModuleNotFoundError",
        root_cause_line=1,
        message="No module named 'pandas'",
        context=["1: import pandas"],
    )

    assert classifier.extract_missing_module(diagnosis) == "pandas"


def test_classifier_classify_from_result_unknown() -> None:
    """classify_from_result falls back to UnknownError when no traceback exists."""
    classifier = TracebackClassifier()

    diagnosis = classifier.classify_from_result("print('ok')", "plain error message")

    assert diagnosis.error_type == "UnknownError"
    assert diagnosis.message == "plain error message"


def test_classifier_context_extraction() -> None:
    """The classifier returns surrounding source lines."""
    classifier = TracebackClassifier(context_radius=1)
    code = "line one\nline two\nline three\nline four"

    context = classifier._extract_context(code, 2)

    assert any("line one" in ctx for ctx in context)
    assert any("line two" in ctx for ctx in context)
    assert any("line three" in ctx for ctx in context)


def test_environment_uv_backend_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """SandboxManager raises when uv is requested but not on PATH."""
    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    sandbox = SandboxManager(backend="uv")
    with pytest.raises(SandboxError):
        sandbox.create()
    sandbox.cleanup()


def test_environment_write_requirements_file() -> None:
    """SandboxManager can write a requirements file."""
    sandbox = SandboxManager(backend="venv")
    with sandbox:
        path = sandbox.write_requirements_file(["tenacity", "pydantic"])
        assert path.exists()
        assert "tenacity" in path.read_text()


def test_environment_cleanup_keep() -> None:
    """SandboxManager keeps the workspace when keep=True."""
    sandbox = SandboxManager(backend="venv", keep=True)
    with sandbox:
        workspace = sandbox.workspace

    assert workspace.exists()


def test_runner_installs_requirements(
    sandbox: SandboxManager,
    runner: SubprocessRunner,
) -> None:
    """SubprocessRunner installs requested packages in the sandbox."""
    sandbox.create()
    try:
        request = CodeExecutionRequest(
            code="import tenacity; print(tenacity.__name__)",
            requirements=["tenacity"],
            timeout_seconds=120,
        )
        result = runner.run(request)

        assert result.exit_code == 0
        assert result.stdout.strip()
    finally:
        sandbox.cleanup()


def test_self_healing_import_error_returns_diagnosis() -> None:
    """An unresolved ModuleNotFoundError returns a diagnosis after retries."""
    engine = SelfHealingEngine(use_llm=False)
    with engine:
        result, diagnosis = engine.run("import not_a_real_module_xyz")

        assert result.exit_code != 0
        assert diagnosis is not None
        assert diagnosis.error_type == "ModuleNotFoundError"


def test_self_healing_syntax_error() -> None:
    """A syntax error returns a synthetic execution result."""
    engine = SelfHealingEngine(use_llm=False)
    with engine:
        result, diagnosis = engine.run("def foo(")

        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr
        assert diagnosis is None


def test_self_healing_max_iterations_validation() -> None:
    """SelfHealingEngine rejects invalid max-iteration counts."""
    with pytest.raises(SelfHealingError):
        SelfHealingEngine(max_iterations=0)


def test_gemini_patch_generator_unconfigured() -> None:
    """GeminiPatchGenerator returns None when no API key is available."""
    generator = GeminiPatchGenerator(api_key=None)
    diagnosis = ErrorDiagnosis(
        error_type="TypeError",
        root_cause_line=1,
        message="unsupported operand type(s)",
        context=["1: print('x' + 1)"],
    )

    assert generator.is_configured is False
    assert generator("print('x' + 1)", diagnosis) is None


def test_main_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package __main__ entry point invokes the CLI and exits cleanly."""
    monkeypatch.setattr(sys, "argv", ["codeshield", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("codeshield", run_name="__main__")

    assert exc_info.value.code == 0


def test_main_module_help_subprocess() -> None:
    """python -m codeshield --help exits successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "codeshield", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "codeshield" in result.stdout


def test_gemini_build_prompt() -> None:
    """The prompt builder includes the diagnosis and the failing code."""
    diagnosis = ErrorDiagnosis(
        error_type="NameError",
        root_cause_line=2,
        message="name 'x' is not defined",
        context=["2: print(x)"],
    )

    prompt = GeminiPatchGenerator._build_prompt("print(x)", diagnosis)

    assert "NameError" in prompt
    assert "print(x)" in prompt
    assert "2: print(x)" in prompt


def test_gemini_extract_code_variants() -> None:
    """_extract_code handles python, plain and raw markdown fences."""
    assert "import math" in GeminiPatchGenerator._extract_code(
        "```python\nimport math\n```"
    )
    assert "import math" in GeminiPatchGenerator._extract_code(
        "```\nimport math\n```"
    )
    assert "import math" in GeminiPatchGenerator._extract_code("import math")


def test_gemini_patch_generator_returns_valid_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid Gemini response is extracted, validated and returned."""
    generator = GeminiPatchGenerator(api_key="fake")
    diagnosis = ErrorDiagnosis(
        error_type="NameError",
        root_cause_line=1,
        message="name 'x' is not defined",
        context=[],
    )

    def _fake_call(_self: GeminiPatchGenerator, _code: str, _diagnosis: ErrorDiagnosis) -> str:
        return "```python\nx = 1\nprint(x)\n```"

    monkeypatch.setattr(GeminiPatchGenerator, "_call_gemini", _fake_call)

    patched = generator("print(x)", diagnosis)

    assert patched is not None
    assert "x = 1" in patched


def test_gemini_patch_generator_rejects_invalid_ast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Gemini response that fails AST validation is rejected."""
    generator = GeminiPatchGenerator(api_key="fake", max_llm_retries=1)
    diagnosis = ErrorDiagnosis(
        error_type="NameError",
        root_cause_line=1,
        message="name 'x' is not defined",
        context=[],
    )

    monkeypatch.setattr(GeminiPatchGenerator, "_call_gemini", lambda _s, _c, _d: "def foo(")

    patched = generator("print(x)", diagnosis)

    assert patched is None


def test_gemini_patch_generator_retries_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API failures are retried up to max_llm_retries."""
    generator = GeminiPatchGenerator(api_key="fake", max_llm_retries=2)
    diagnosis = ErrorDiagnosis(
        error_type="TypeError",
        root_cause_line=1,
        message="unsupported operand",
        context=[],
    )
    call_count = 0

    def _failing_call(_self: GeminiPatchGenerator, _code: str, _diagnosis: ErrorDiagnosis) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("API down")

    monkeypatch.setattr(GeminiPatchGenerator, "_call_gemini", _failing_call)

    patched = generator("print('x' + 1)", diagnosis)

    assert patched is None
    assert call_count == 2


def test_gemini_call_gemini_with_fake_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_call_gemini builds the client and returns the raw response."""
    pytest.importorskip("google.genai")
    from google import genai
    from google.genai import types as genai_types

    fake_client_class = MagicMock()
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "x = 1\nprint(x)"
    fake_client.models.generate_content.return_value = fake_response
    fake_client_class.return_value = fake_client

    fake_part = MagicMock()
    fake_part.from_text = MagicMock(return_value=fake_part)

    monkeypatch.setattr(genai, "Client", fake_client_class)
    monkeypatch.setattr(genai_types, "Part", fake_part)
    monkeypatch.setattr(genai_types, "Content", MagicMock())
    monkeypatch.setattr(genai_types, "GenerateContentConfig", MagicMock())

    generator = GeminiPatchGenerator(api_key="fake")
    diagnosis = ErrorDiagnosis(
        error_type="NameError",
        root_cause_line=1,
        message="name 'x' is not defined",
        context=[],
    )

    raw = generator._call_gemini("print(x)", diagnosis)

    assert "x = 1" in raw
    fake_client_class.assert_called_once_with(api_key="fake")

