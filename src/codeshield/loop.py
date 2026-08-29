"""Self-healing execution loop."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from codeshield.analyzer import validate_syntax_and_safety
from codeshield.classifier import TracebackClassifier
from codeshield.environment import SandboxManager
from codeshield.runner import SubprocessRunner
from codeshield.schemas import (
    CodeExecutionRequest,
    ErrorDiagnosis,
    ExecutionResult,
    PatchProposal,
    ValidationReport,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_LLM_RETRIES = 3
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class SelfHealingError(RuntimeError):
    """Raised when the self-healing loop cannot complete execution safely."""


class LLMPatchError(RuntimeError):
    """Raised when the LLM patch generator cannot produce a safe correction."""


class GeminiPatchGenerator:
    """Generate patches using the Google Gemini API with AST validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_llm_retries: int = DEFAULT_MAX_LLM_RETRIES,
    ) -> None:
        """Initialize the Gemini patch generator.

        Args:
            api_key: Gemini API key. If ``None``, ``GEMINI_API_KEY`` env var is used.
            model: Gemini model name. Defaults to ``GEMINI_MODEL`` env var or
                ``gemini-2.5-flash``.
            max_llm_retries: Maximum attempts to ask Gemini for a valid patch.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self._max_llm_retries = max(1, max_llm_retries)

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when an API key is available."""
        return bool(self._api_key)

    def __call__(self, code: str, diagnosis: ErrorDiagnosis) -> str | None:
        """Generate a patched version of ``code`` using Gemini, or ``None``."""
        if not self.is_configured:
            return None

        for attempt in range(1, self._max_llm_retries + 1):
            logger.info(
                "Requesting patch from Gemini (attempt %d/%d)",
                attempt,
                self._max_llm_retries,
            )
            try:
                raw_response = self._call_gemini(code, diagnosis)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini API call failed: %s", exc)
                continue

            if not raw_response:
                continue

            patched_code = self._extract_code(raw_response)
            if patched_code == code or not patched_code.strip():
                continue

            report = validate_syntax_and_safety(patched_code)
            if report.is_valid:
                logger.info("Gemini patch passed AST validation")
                return patched_code

            logger.warning(
                "Gemini patch failed AST validation: %s",
                report.violations,
            )

        logger.warning("Gemini could not produce a valid patch after all retries")
        return None

    def _call_gemini(self, code: str, diagnosis: ErrorDiagnosis) -> str:
        """Call the Gemini API and return the raw text response."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMPatchError(
                "google-genai is not installed. "
                "Install it with: pip install 'autonomous-code-execution-engine[llm]'"
            ) from exc

        prompt = self._build_prompt(code, diagnosis)
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        if not response or not response.text:
            return ""
        return response.text

    @staticmethod
    def _build_prompt(code: str, diagnosis: ErrorDiagnosis) -> str:
        """Build a deterministic prompt for Gemini."""
        context = "\n".join(diagnosis.context) if diagnosis.context else "N/A"
        return (
            "You are an expert Python debugger. "
            "Given the code, the runtime error and the surrounding context, "
            "return ONLY the corrected Python code. "
            "Do not include explanations, comments or markdown formatting.\n\n"
            f"Error type: {diagnosis.error_type}\n"
            f"Error message: {diagnosis.message}\n"
            f"Failing line: {diagnosis.root_cause_line}\n"
            f"Context:\n{context}\n\n"
            f"Code:\n{code}\n\n"
            "Corrected code:"
        )

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Extract a Python code block from a markdown-wrapped response."""
        fenced = re.search(r"```python\n(.*?)\n```", raw, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        plain = re.search(r"```\n(.*?)\n```", raw, re.DOTALL)
        if plain:
            return plain.group(1).strip()

        return raw.strip()


class SelfHealingEngine:
    """Orchestrate AST validation, sandboxed execution and self-healing retries."""

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        sandbox: SandboxManager | None = None,
        runner: SubprocessRunner | None = None,
        classifier: TracebackClassifier | None = None,
        patch_generator: Callable[[str, ErrorDiagnosis], str | None] | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        use_llm: bool = True,
    ) -> None:
        """Initialize the self-healing engine.

        Args:
            max_iterations: Maximum number of validation/execution iterations.
            sandbox: Optional ``SandboxManager``; a temporary one is created when
                ``None``.
            runner: Optional ``SubprocessRunner``; built from ``sandbox`` when ``None``.
            classifier: Optional ``TracebackClassifier``.
            patch_generator: Optional callable ``(code, diagnosis) -> patched_code``
                used to produce corrections. Overrides the Gemini/local default.
            gemini_api_key: Optional Gemini API key. Falls back to ``GEMINI_API_KEY``
                environment variable.
            gemini_model: Optional Gemini model name. Falls back to ``GEMINI_MODEL``
                environment variable or ``gemini-2.5-flash``.
            use_llm: If ``True`` and a Gemini API key is available, use Gemini for
                patch generation; otherwise fall back to the local deterministic
                generator.
        """
        if max_iterations < 1:
            raise SelfHealingError("max_iterations must be at least 1")

        self._max_iterations = max_iterations
        self._sandbox = sandbox or SandboxManager()
        self._runner = runner or SubprocessRunner(self._sandbox)
        self._classifier = classifier or TracebackClassifier()

        if patch_generator is not None:
            self._patch_generator = patch_generator
        elif use_llm:
            gemini = GeminiPatchGenerator(
                api_key=gemini_api_key,
                model=gemini_model,
            )
            self._patch_generator = (
                gemini if gemini.is_configured else self._default_patch_generator
            )
        else:
            self._patch_generator = self._default_patch_generator

    def run(
        self,
        request: CodeExecutionRequest | str,
        timeout: float | None = None,
    ) -> tuple[ExecutionResult, ErrorDiagnosis | None]:
        """Run code through the AST -> sandbox -> heal loop.

        Args:
            request: Either a ``CodeExecutionRequest`` or a raw source string.
            timeout: Optional execution timeout override.

        Returns:
            The final ``ExecutionResult`` and an optional ``ErrorDiagnosis``.

        Raises:
            SelfHealingError: when the loop exhausts all iterations without a
                clean result.
        """
        if isinstance(request, str):
            request = CodeExecutionRequest(code=request)

        diagnosis: ErrorDiagnosis | None = None
        for attempt in range(1, self._max_iterations + 1):
            logger.info("Self-healing iteration %d/%d", attempt, self._max_iterations)

            report = validate_syntax_and_safety(request.code)
            if not report.is_valid:
                if report.exception is not None:
                    return self._syntax_failure_result(request, report), None

                diagnosis = ErrorDiagnosis(
                    error_type="StaticSafetyViolation",
                    root_cause_line=None,
                    message="; ".join(report.violations),
                    context=[],
                )
                patched = self._generate_and_validate_patch(request.code, diagnosis)
                if patched is None:
                    raise SelfHealingError(
                        f"Static safety violations cannot be auto-patched: {report.violations}"
                    )
                request = self._apply_patch(request, patched, attempt)
                continue

            result = self._runner.run(request, timeout=timeout)

            if self._is_success(result):
                return result, None

            diagnosis = self._classifier.classify_from_result(request.code, result.stderr)
            patched = self._generate_and_validate_patch(request.code, diagnosis)

            if patched is None:
                logger.warning(
                    "No deterministic patch available for %s at line %s",
                    diagnosis.error_type,
                    diagnosis.root_cause_line,
                )
                return result, diagnosis

            request = self._apply_patch(request, patched, attempt)

        raise SelfHealingError(
            f"Self-healing loop exhausted after {self._max_iterations} attempts. "
            f"Last diagnosis: {diagnosis.error_type}"
        )

    def _is_success(self, result: ExecutionResult) -> bool:
        """Return ``True`` when the result represents a clean execution."""
        return (
            result.exit_code == 0
            and not result.silent_failure_detected
            and not result.timed_out
        )

    def _generate_and_validate_patch(
        self,
        code: str,
        diagnosis: ErrorDiagnosis,
    ) -> PatchProposal | None:
        """Produce a patch, validate it with the AST and return the proposal."""
        patched_code = self._patch_generator(code, diagnosis)
        if patched_code is None or patched_code == code:
            return None

        report = validate_syntax_and_safety(patched_code)
        return PatchProposal(
            file_path=Path("<dynamic>"),
            patched_code=patched_code,
            is_syntax_valid=report.is_valid,
            diagnosis=diagnosis,
        )

    def _apply_patch(
        self,
        request: CodeExecutionRequest,
        proposal: PatchProposal,
        attempt: int,
    ) -> CodeExecutionRequest:
        """Return a new request with the patched code and an updated file name."""
        if not proposal.is_syntax_valid:
            raise SelfHealingError(
                f"Proposed patch failed AST validation: {proposal.patched_code[:200]}"
            )

        file_name = f"script_attempt_{attempt}.py"
        return CodeExecutionRequest(
            code=proposal.patched_code,
            timeout_seconds=request.timeout_seconds,
            requirements=list(request.requirements),
            file_name=file_name,
        )

    def _syntax_failure_result(
        self,
        request: CodeExecutionRequest,
        report: ValidationReport,
    ) -> ExecutionResult:
        """Build a synthetic ``ExecutionResult`` for a syntax validation failure."""
        stderr = report.violations[0] if report.violations else "Syntax validation failed"
        return ExecutionResult(
            stdout="",
            stderr=stderr,
            exit_code=1,
            duration_seconds=0.0,
            silent_failure_detected=False,
            timed_out=False,
        )

    def _default_patch_generator(
        self,
        code: str,
        diagnosis: ErrorDiagnosis,
    ) -> str | None:
        """Generate a deterministic patch based on the diagnosis.

        The default generator only handles a small, safe subset of runtime errors:
        - ``NameError``: add an ``import`` or placeholder definition.
        - ``ImportError`` / ``ModuleNotFoundError``: try a well-known alias.
        - ``SyntaxError``: not auto-patched (returns ``None``).

        More complex errors require a model-based patch generator.
        """
        if diagnosis.error_type == "NameError":
            missing = self._classifier.extract_missing_name(diagnosis)
            if missing is None:
                return None

            if missing in {
                "math",
                "json",
                "os",
                "sys",
                "re",
                "time",
                "datetime",
                "collections",
                "itertools",
                "pathlib",
                "typing",
            }:
                return f"import {missing}\n{code}"
            return f"{missing} = None\n{code}"

        if diagnosis.error_type in {"ImportError", "ModuleNotFoundError"}:
            missing = self._classifier.extract_missing_module(diagnosis)
            if missing is None:
                return None

            # Simple alias fallback for common data-science package names.
            aliases: dict[str, str] = {
                "sklearn": "import sklearn",
                "pandas": "import pandas as pd",
                "numpy": "import numpy as np",
                "matplotlib": "import matplotlib",
            }
            if missing in aliases:
                return f"{aliases[missing]}\n{code}"
            return None

        if diagnosis.error_type in {
            "IndexError",
            "TypeError",
            "AttributeError",
            "ZeroDivisionError",
            "ValueError",
            "KeyError",
        }:
            # These categories require domain knowledge to patch safely.
            # Gemini handles them when the LLM path is configured.
            return None

        return None

    def __enter__(self) -> SelfHealingEngine:
        """Ensure the sandbox exists when used as a context manager."""
        self._sandbox.create()
        return self

    def __exit__(self, *exc: object) -> None:
        self._sandbox.cleanup()
