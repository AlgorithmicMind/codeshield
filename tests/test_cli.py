"""Functional tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from execution_engine.cli import main


def test_cli_runs_clean_file(tmp_path: Path) -> None:
    """The CLI executes a valid Python file and exits with 0."""
    script = tmp_path / "clean.py"
    script.write_text("print('ok')")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "10"])

    assert exit_code == 0


def test_cli_runs_with_llm_enabled_no_key(tmp_path: Path) -> None:
    """The CLI defaults to the local fallback when no Gemini API key is set."""
    script = tmp_path / "name_error.py"
    script.write_text("print(math.pi)")

    exit_code = main(["run", str(script), "--llm", "--timeout", "30"])

    assert exit_code == 0


def test_cli_runs_with_llm_disabled_fallback(tmp_path: Path) -> None:
    """The --no-llm flag forces the deterministic fallback patch generator."""
    script = tmp_path / "name_error.py"
    script.write_text("print(math.pi)")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "30"])

    assert exit_code == 0


def test_cli_missing_file() -> None:
    """The CLI exits with 1 when the supplied file does not exist."""
    exit_code = main(["run", "nonexistent_file_xyz.py"])

    assert exit_code == 1


def test_cli_no_command() -> None:
    """The CLI exits with 2 when no subcommand is supplied."""
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2


def test_cli_help() -> None:
    """The --help flag exits with 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_cli_read_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CLI exits with 1 when the file cannot be read."""
    script = tmp_path / "unreadable.py"
    script.write_text("print('ok')")

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise OSError("cannot read file")

    monkeypatch.setattr(Path, "read_text", _raise)

    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(script)])

    assert exc_info.value.code == 1


def test_cli_execution_with_stderr(tmp_path: Path) -> None:
    """The CLI prints stderr and exits with 0 when only warnings are emitted."""
    script = tmp_path / "warnings.py"
    script.write_text("import sys; sys.stderr.write('warning\\n')")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "10"])

    assert exit_code == 0


def test_cli_silent_failure_returns_one(tmp_path: Path) -> None:
    """The CLI exits with 1 when a silent failure is detected."""
    script = tmp_path / "silent.py"
    script.write_text("print('empty DataFrame')")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "10"])

    assert exit_code == 1


def test_cli_timeout_returns_one(tmp_path: Path) -> None:
    """The CLI exits with 1 when the execution times out."""
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(60)")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "1"])

    assert exit_code == 1


def test_cli_diagnosis_returns_one(tmp_path: Path) -> None:
    """The CLI exits with 1 and reports the diagnosis for a runtime error."""
    script = tmp_path / "broken.py"
    script.write_text("import not_a_real_module_xyz")

    exit_code = main(["run", str(script), "--no-llm", "--timeout", "10"])

    assert exit_code == 1
