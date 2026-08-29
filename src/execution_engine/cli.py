"""Command-line interface for the execution engine."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from execution_engine.loop import SelfHealingEngine

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="execution_engine",
        description="Deterministic, isolated, and self-healing Python code execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a Python source file inside a sandbox.",
    )
    run_parser.add_argument(
        "file",
        type=Path,
        help="Path to the Python file to execute.",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Execution timeout in seconds (default: 60).",
    )
    run_parser.add_argument(
        "--llm",
        action="store_true",
        default=True,
        help="Enable LLM-guided self-healing when GEMINI_API_KEY is set (default: on).",
    )
    run_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-guided self-healing and use the local fallback.",
    )

    return parser


def _read_file(file_path: Path) -> str:
    """Read and return the contents of ``file_path``."""
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Could not read %s: %s", file_path, exc)
        raise SystemExit(1) from exc


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 2

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        return 1

    code = _read_file(args.file)
    use_llm = not args.no_llm if args.no_llm else args.llm

    engine = SelfHealingEngine(use_llm=use_llm)
    with engine:
        result, diagnosis = engine.run(code, timeout=args.timeout)

    print("--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

    if result.timed_out:
        print("Execution timed out")
    if result.silent_failure_detected:
        print("Silent failure detected")
    if diagnosis:
        print(f"Diagnosis: {diagnosis.error_type} - {diagnosis.message}")

    return 0 if result.exit_code == 0 and not result.silent_failure_detected else 1


if __name__ == "__main__":
    sys.exit(main())
