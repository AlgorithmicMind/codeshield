"""Example 3: full self-healing workflow with optional Gemini support."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from execution_engine.loop import SelfHealingEngine


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        code = 'print("Result: " + 42)'
        print("Gemini API key found: running LLM-guided self-healing")
    else:
        code = "print(undefined_value)"
        print("Gemini API key not found; running local fallback demo")

    print(f"\n1) Original broken code:\n   {code!r}")
    print("\n2) Running the self-healing engine...")

    engine = SelfHealingEngine()
    with engine:
        result, diagnosis = engine.run(code)

    print("\n3) Repaired execution:")
    print(f"   Exit code: {result.exit_code}")
    print(f"   Stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"   Stderr: {result.stderr.strip()}")
    if result.timed_out:
        print("   Execution timed out")
    if result.silent_failure_detected:
        print("   Silent failure detected")
    if diagnosis:
        print(f"   Diagnosis: {diagnosis.error_type} - {diagnosis.message}")


if __name__ == "__main__":
    main()
