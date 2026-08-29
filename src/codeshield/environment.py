"""Management of isolated ``uv`` virtual environments used as sandboxes."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Raised when a sandbox cannot be created, destroyed or updated."""


class SandboxManager:
    """Create, populate and clean up an isolated virtual environment.

    The manager prefers the ``uv`` toolchain when available, and transparently
    falls back to the standard ``venv``/``pip`` tooling otherwise. This makes
    the package usable on systems where ``uv`` has not yet been installed while
    still taking advantage of ``uv`` when it is present.
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        venv_name: str = ".venv",
        python: str | None = None,
        keep: bool = False,
        backend: str | None = None,
    ) -> None:
        """Initialize a sandbox manager.

        Args:
            workspace: Directory that will host the virtual environment. If ``None``,
                a temporary directory is created.
            venv_name: Name of the virtual environment directory inside ``workspace``.
            python: Python version or interpreter requested for the backend.
            keep: If ``False`` (default), the workspace is removed when the manager
                is garbage collected or ``cleanup`` is called.
            backend: Environment backend to use. ``"uv"`` or ``"venv"``. When ``None``,
                ``uv`` is auto-detected and preferred.
        """
        self._workspace = (
            Path(workspace)
            if workspace
            else Path(tempfile.mkdtemp(prefix="codeshield_"))
        )
        self._venv_name = venv_name
        self._python = python or sys.executable
        self._keep = keep
        self._backend = self._resolve_backend(backend)
        self._uv_path: Path | None = None
        self._venv_path = self._workspace / self._venv_name

    @property
    def workspace(self) -> Path:
        """Return the root sandbox workspace directory."""
        return self._workspace

    @property
    def workspace_path(self) -> Path:
        """Alias for the root sandbox workspace directory."""
        return self._workspace

    @property
    def venv_path(self) -> Path:
        """Return the virtual environment directory."""
        return self._venv_path

    @property
    def python_executable(self) -> Path:
        """Return the interpreter path inside the virtual environment."""
        if self._is_windows():
            return self._venv_path / "Scripts" / "python.exe"
        return self._venv_path / "bin" / "python"

    def uv_executable(self) -> Path:
        """Resolve and cache the ``uv`` executable path.

        Raises:
            SandboxError: when ``uv`` is not available on ``PATH``.
        """
        if self._uv_path is None:
            uv = shutil.which("uv")
            if uv is None:
                raise SandboxError(
                    "The `uv` executable was not found on PATH. "
                    "Install uv from https://github.com/astral-sh/uv"
                )
            self._uv_path = Path(uv)
        return self._uv_path

    def create(self) -> Path:
        """Create the virtual environment.

        Returns:
            The path to the virtual environment.

        Raises:
            SandboxError: if the backend fails to create the environment.
        """
        self._workspace.mkdir(parents=True, exist_ok=True)

        if self._backend == "uv":
            self._create_with_uv()
        else:
            self._create_with_venv()

        if not self.python_executable.exists():
            raise SandboxError(
                f"Virtual environment was created but interpreter not found at "
                f"{self.python_executable}"
            )

        logger.info("Virtual environment ready at %s", self._venv_path)
        return self._venv_path

    def _create_with_uv(self) -> None:
        """Create a virtual environment using ``uv venv``."""
        cmd: list[str] = [str(self.uv_executable()), "venv"]
        if self._python:
            cmd.extend(["--python", self._python])
        cmd.append(str(self._venv_name))

        logger.info("Creating uv venv at %s", self._venv_path)
        result = self._run_command(cmd, cwd=self._workspace)
        if result.returncode != 0:
            raise SandboxError(
                f"`uv venv` failed (exit {result.returncode}): "
                f"{result.stderr or result.stdout}"
            )

    def _create_with_venv(self) -> None:
        """Create a virtual environment using the standard ``venv`` module."""
        cmd = [self._python, "-m", "venv", str(self._venv_name)]
        logger.info("Creating venv with %s at %s", self._python, self._venv_path)
        result = self._run_command(cmd, cwd=self._workspace)
        if result.returncode != 0:
            raise SandboxError(
                f"`python -m venv` failed (exit {result.returncode}): "
                f"{result.stderr or result.stdout}"
            )

    def install_requirements(self, requirements: Sequence[str] | Path | str) -> None:
        """Install packages from a list or a ``requirements.txt`` path.

        Args:
            requirements: Either a sequence of package specifiers or a path
                to a requirements file.

        Raises:
            SandboxError: if the installation fails.
        """
        if isinstance(requirements, (str, Path)):
            requirements_path = Path(requirements)
            if not requirements_path.exists():
                raise SandboxError(f"Requirements file not found: {requirements_path}")

            if self._backend == "uv":
                cmd = [
                    str(self.uv_executable()),
                    "pip",
                    "install",
                    "-r",
                    str(requirements_path),
                ]
            else:
                cmd = [
                    str(self.python_executable),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(requirements_path),
                ]
        else:
            if not requirements:
                return
            if self._backend == "uv":
                cmd = [str(self.uv_executable()), "pip", "install", *requirements]
            else:
                cmd = [str(self.python_executable), "-m", "pip", "install", *requirements]

        logger.info("Installing requirements in sandbox: %s", cmd)
        result = self._run_command(cmd, cwd=self._workspace)
        if result.returncode != 0:
            raise SandboxError(
                f"Requirement installation failed (exit {result.returncode}): "
                f"{result.stderr or result.stdout}"
            )

    def write_requirements_file(
        self, packages: Sequence[str], file_name: str = "requirements.txt"
    ) -> Path:
        """Write package specifiers to ``workspace/requirements.txt``.

        Returns:
            Path of the written file.
        """
        path = self._workspace / file_name
        path.write_text("\n".join(packages) + "\n", encoding="utf-8")
        return path

    def cleanup(self) -> None:
        """Remove the workspace unless ``keep`` was set to ``True``."""
        if self._keep or not self._workspace.exists():
            return
        logger.info("Cleaning up sandbox workspace: %s", self._workspace)
        try:
            shutil.rmtree(self._workspace, ignore_errors=True)
        except OSError as exc:
            logger.warning("Could not remove workspace %s: %s", self._workspace, exc)

    def __enter__(self) -> SandboxManager:
        self.create()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def _resolve_backend(self, backend: str | None) -> str:
        """Resolve the environment backend, preferring ``uv`` when available."""
        if backend in {"uv", "venv"}:
            return backend
        if backend is not None:
            raise SandboxError(f"Unknown backend '{backend}'; choose 'uv' or 'venv'")
        if shutil.which("uv") is not None:
            return "uv"
        logger.warning(
            "uv was not found on PATH; falling back to the standard venv/pip backend. "
            "Install uv from https://github.com/astral-sh/uv for faster sandboxes."
        )
        return "venv"

    def _run_command(
        self,
        cmd: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command and return a ``CompletedProcess`` with text output."""
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise SandboxError(f"Failed to spawn command {cmd[0]!r}: {exc}") from exc

    @staticmethod
    def _is_windows() -> bool:
        """Return ``True`` when running on Windows."""
        return sys.platform.startswith("win")
