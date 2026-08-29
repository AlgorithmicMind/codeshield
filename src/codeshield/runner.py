"""Subprocess runner that executes Python code inside a ``uv`` sandbox."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterable
from typing import IO

from codeshield.environment import SandboxError, SandboxManager
from codeshield.schemas import CodeExecutionRequest, ExecutionResult

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_SILENT_PATTERNS: tuple[str, ...] = (
    r"empty\s+[Dd]ata[Ff]rame",
    r"all\s+[Nn]a[Nn]",
    r"Traceback",
    r"Pipeline\s+failed",
    r"[Ff]atal\s+[Ee]rror",
)


class SubprocessRunnerError(RuntimeError):
    """Raised when the runner cannot prepare or launch a code execution."""


class SubprocessRunner:
    """Execute Python source code in an isolated subprocess with streaming I/O."""

    def __init__(
        self,
        sandbox: SandboxManager,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        silent_failure_patterns: Iterable[str] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            sandbox: ``SandboxManager`` that provides the isolated interpreter.
            default_timeout: Default execution timeout in seconds.
            silent_failure_patterns: Optional regex patterns used to detect silent
                failures inside process output.
        """
        if default_timeout <= 0:
            raise SubprocessRunnerError("default_timeout must be greater than 0")

        self._sandbox = sandbox
        self._default_timeout = default_timeout
        self._silent_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (silent_failure_patterns or DEFAULT_SILENT_PATTERNS)
        ]

    def run(
        self,
        request: CodeExecutionRequest | str,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Run ``request.code`` inside the sandbox and return the result.

        Args:
            request: Either a ``CodeExecutionRequest`` or a raw source string.
            timeout: Optional override for the execution timeout. Defaults to
                ``default_timeout`` or ``request.timeout_seconds`` when a request
                object is supplied.

        Returns:
            An ``ExecutionResult`` with captured output, timing and flags.

        Raises:
            SubprocessRunnerError: when the sandbox interpreter is missing.
        """
        if isinstance(request, str):
            request = CodeExecutionRequest(code=request)

        effective_timeout = timeout if timeout is not None else request.timeout_seconds
        file_name = request.file_name or "script.py"
        script_path = self._sandbox.workspace / file_name

        try:
            script_path.write_text(request.code, encoding="utf-8")
        except OSError as exc:
            raise SubprocessRunnerError(f"Failed to write script to {script_path}: {exc}") from exc

        if not self._sandbox.python_executable.exists():
            raise SubprocessRunnerError(
                f"Sandbox interpreter not found at {self._sandbox.python_executable}"
            )

        self._install_requirements_if_needed(request.requirements)

        cmd = [str(self._sandbox.python_executable), str(script_path)]
        env = self._build_environment()

        logger.info("Executing %s with timeout %.1fs", script_path, effective_timeout)
        return self._run_subprocess(cmd, env, effective_timeout)

    def _install_requirements_if_needed(self, packages: list[str]) -> None:
        """Install requested packages in the sandbox before execution."""
        if not packages:
            return
        try:
            self._sandbox.install_requirements(packages)
        except SandboxError as exc:
            raise SubprocessRunnerError(f"Failed to install requirements: {exc}") from exc

    def _build_environment(self) -> dict[str, str]:
        """Return the environment for the subprocess.

        Inherit ``PATH`` from the parent so that ``uv``-managed binaries work,
        but remove Python-specific variables that could leak the parent venv.
        """
        env = os.environ.copy()
        for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _run_subprocess(
        self,
        cmd: list[str],
        env: dict[str, str],
        timeout: float,
    ) -> ExecutionResult:
        """Launch the process, stream I/O, enforce timeout and return result."""
        start = time.monotonic()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
        except OSError as exc:
            raise SubprocessRunnerError(f"Failed to start subprocess: {exc}") from exc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def reader(pipe: IO[str], sink: list[str]) -> None:
            """Read lines from a pipe without blocking the main thread."""
            try:
                for line in pipe:
                    sink.append(line)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stream reader terminated: %s", exc)
            finally:
                pipe.close()

        stdout_thread = threading.Thread(
            target=reader,
            args=(process.stdout, stdout_lines),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=reader,
            args=(process.stderr, stderr_lines),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning("Process exceeded timeout %.1fs; terminating", timeout)
            self._terminate_process(process)

        # Give the reader threads a short grace period to flush remaining bytes.
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)

        duration = time.monotonic() - start
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        exit_code = process.returncode
        if exit_code is None:
            exit_code = -1

        silent_failure = self._detect_silent_failure(exit_code, stdout, stderr)

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=round(duration, 6),
            silent_failure_detected=silent_failure,
            timed_out=timed_out,
        )

    def _terminate_process(self, process: subprocess.Popen) -> None:
        """Gracefully terminate and, if necessary, kill the process."""
        with contextlib.suppress(OSError):
            process.terminate()

        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                process.kill()
            process.wait(timeout=2.0)

    def _detect_silent_failure(self, exit_code: int, stdout: str, stderr: str) -> bool:
        """Return ``True`` when a zero exit code hides suspicious output patterns."""
        if exit_code != 0:
            return False

        combined = f"{stdout}\n{stderr}"
        return any(pattern.search(combined) for pattern in self._silent_patterns)
