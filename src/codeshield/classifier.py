"""Traceback parsing and error classification utilities."""

from __future__ import annotations

import logging
import re

from codeshield.schemas import ErrorDiagnosis

logger = logging.getLogger(__name__)


class TracebackClassifier:
    """Extract structured diagnostics from Python tracebacks."""

    def __init__(self, context_radius: int = 2) -> None:
        """Initialize the classifier.

        Args:
            context_radius: Number of source lines to include before and after
                the failing line.
        """
        self._context_radius = max(0, context_radius)

    def classify(self, code: str, stderr: str) -> ErrorDiagnosis | None:
        """Extract an ``ErrorDiagnosis`` from ``stderr``.

        Args:
            code: Original source code used to provide context.
            stderr: Standard error produced by the executed process.

        Returns:
            An ``ErrorDiagnosis`` if a traceback could be parsed, otherwise ``None``.
        """
        if not stderr.strip():
            return None

        if "Traceback" not in stderr:
            return None

        exception_match = self._extract_exception_line(stderr)
        if not exception_match:
            return None

        exc_type, exc_message = exception_match
        line_no = self._extract_failing_line(stderr, code)
        context = self._extract_context(code, line_no)

        logger.debug(
            "Classified %s at line %s: %s",
            exc_type,
            line_no,
            exc_message,
        )
        return ErrorDiagnosis(
            error_type=exc_type,
            root_cause_line=line_no,
            message=exc_message,
            context=context,
        )

    def classify_from_result(self, code: str, stderr: str) -> ErrorDiagnosis:
        """Return a diagnosis, falling back to a generic one when parsing fails."""
        diagnosis = self.classify(code, stderr)
        if diagnosis is not None:
            return diagnosis

        return ErrorDiagnosis(
            error_type="UnknownError",
            root_cause_line=None,
            message=stderr.strip() or "An unknown error occurred during execution.",
            context=[],
        )

    def _extract_exception_line(self, stderr: str) -> tuple[str, str] | None:
        """Parse the final traceback line ``ExceptionName: message``.

        Returns:
            A tuple ``(exception_name, message)`` or ``None``.
        """
        # Find the last line starting after the traceback header that matches
        # an exception declaration. We scan all lines and keep the last valid one.
        in_traceback = False
        last_match: tuple[str, str] | None = None

        for raw_line in stderr.splitlines():
            line = raw_line.rstrip()
            if in_traceback:
                pattern = r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):\s*(.*)$"
                match = re.match(pattern, line)
                if match:
                    exception_name = match.group(1)
                    message = match.group(2).strip()
                    # Ignore built-in notes that follow the exception line.
                    last_match = (exception_name, message)
            if line.startswith("Traceback"):
                in_traceback = True

        return last_match

    def _extract_failing_line(self, stderr: str, code: str) -> int | None:
        """Determine the most relevant failing source line from the traceback.

        The method looks for the deepest file in the executed code, and if the
        script is not external, returns its line number. If only external frames
        are available, the last line number is returned as a fallback.
        """
        # Pattern: File "...", line X, in ...
        pattern = re.compile(r'File "([^"]+)", line (\d+), in (.+)')
        lines = code.splitlines()

        last_line: int | None = None
        for raw in reversed(stderr.splitlines()):
            match = pattern.search(raw)
            if not match:
                continue
            file_path, line_no_str, _ = match.groups()
            line_no = int(line_no_str)
            last_line = line_no

            # If the frame references the inline script, use it directly.
            if "<string>" in file_path or file_path.endswith("script.py"):
                return line_no

            # Heuristic: line number is inside the provided source range.
            if 1 <= line_no <= len(lines):
                return line_no

        return last_line

    def _extract_context(self, code: str, line_no: int | None) -> list[str]:
        """Return surrounding source lines for ``line_no``."""
        if line_no is None:
            return []

        lines = code.splitlines()
        if not lines:
            return []

        start = max(1, line_no - self._context_radius)
        end = min(len(lines), line_no + self._context_radius)
        return [f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)]

    def extract_missing_name(self, diagnosis: ErrorDiagnosis) -> str | None:
        """For ``NameError``, try to extract the missing identifier."""
        if diagnosis.error_type != "NameError":
            return None
        match = re.search(r"name '([^']+)' is not defined", diagnosis.message)
        if match:
            return match.group(1)
        return None

    def extract_missing_module(self, diagnosis: ErrorDiagnosis) -> str | None:
        """For ``ImportError``/``ModuleNotFoundError``, try to extract the module name."""
        if diagnosis.error_type not in {"ImportError", "ModuleNotFoundError"}:
            return None
        match = re.search(r"No module named '([^']+)'", diagnosis.message)
        if match:
            return match.group(1)
        return None


__all__: list[str] = ["TracebackClassifier"]
