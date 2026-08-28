"""Demonstrate the self-healing engine with Gemini or local fallback."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from execution_engine.loop import SelfHealingEngine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_gemini_demo() -> None:
    """Run a TypeError example and let Gemini propose a fix."""
    code = 'print("Result: " + 42)'
    print("=" * 60)
    print("GEMINI SELF-HEALING DEMO")
    print("=" * 60)
    print("Original code:")
    print(code)
    print()

    engine = SelfHealingEngine()
    with engine:
        result, diagnosis = engine.run(code)

    print("-" * 60)
    print(f"Exit code: {result.exit_code}")
    print(f"Stdout: {result.stdout.strip()}")
    print(f"Stderr: {result.stderr.strip()}")
    if diagnosis:
        print(f"Diagnosis: {diagnosis.error_type} - {diagnosis.message}")
    print()


def run_fallback_demo() -> None:
    """Run a NameError example with the local deterministic fallback."""
    code = "print(undefined_message)"
    print("=" * 60)
    print("LOCAL FALLBACK SELF-HEALING DEMO")
    print("=" * 60)
    print("Original code:")
    print(code)
    print()

    engine = SelfHealingEngine(use_llm=False)
    with engine:
        result, diagnosis = engine.run(code)

    print("-" * 60)
    print(f"Exit code: {result.exit_code}")
    print(f"Stdout: {result.stdout.strip()}")
    if diagnosis:
        print(f"Diagnosis: {diagnosis.error_type} - {diagnosis.message}")
    print()


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        run_gemini_demo()
    else:
        print("GEMINI_API_KEY not found in environment; running local fallback demo.\n")
        run_fallback_demo()
