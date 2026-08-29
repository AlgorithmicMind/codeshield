"""Example 1: execute Python code in an isolated sandbox and measure timings."""

from __future__ import annotations

import time

from execution_engine.environment import SandboxManager
from execution_engine.runner import SubprocessRunner
from execution_engine.schemas import CodeExecutionRequest


def main() -> None:
    code = """
import math
import time
start = time.time()
result = math.factorial(50)
print(f"50! = {result}")
print(f"duration={time.time() - start:.4f}s")
"""

    sandbox = SandboxManager()
    runner = SubprocessRunner(sandbox)

    with sandbox:
        started = time.perf_counter()
        request = CodeExecutionRequest(code=code, timeout_seconds=30)
        result = runner.run(request)
        wall_time = time.perf_counter() - started

        print(f"Wall time: {wall_time:.2f}s")
        print(f"Exit code: {result.exit_code}")
        print(f"Sandbox duration: {result.duration_seconds}s")
        print("--- stdout ---")
        print(result.stdout)
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr)
        if result.timed_out:
            print("WARNING: execution timed out")


if __name__ == "__main__":
    main()
