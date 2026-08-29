"""Example 4: real-world agentic function calling with CodeShield + Gemini.

Two didactic rounds are executed, printing a staged trace of every hop:

1. **Legitimate Analytics**: the agent reasons, writes a financial computation,
   clears the AST gate and executes it inside the ephemeral uv sandbox.
2. **Security Defense**: the agent is pushed towards a dangerous primitive, the
   AST gate blocks it before execution, and the structured error is fed back to
   the agent so it can rectify with a safe snippet.

Stages printed: ``[USER PROMPT]``, ``[AGENT THOUGHT]``,
``[AGENT DECISION & TOOL CALL]``, ``[CODESHIELD SANDBOX RUNTIME]`` and
``[FINAL AGENT RESPONSE]``.

The function-calling loop is driven manually (automatic function calling is
disabled) so that the intermediate ``candidates[0].content.parts`` -- the
model's reasoning text and its ``function_call`` parts -- can be inspected and
printed stage by stage.

Without ``GEMINI_API_KEY`` the script replays both rounds offline against the
very same sandbox using pre-recorded snippets, so it never fails in CI.
"""

import os
import sys
import time
from collections.abc import Callable

from dotenv import load_dotenv

from codeshield import create_code_execution_tool, validate_syntax_and_safety

try:  # ``google-genai`` ships in the optional ``llm`` extra.
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

MAX_TURNS = 5

ROUND_1_PROMPT = (
    "Calculate the cumulative return and the annualized Sharpe Ratio "
    "(assuming a risk-free rate of 0.02) for this daily return series:\n"
    "[0.012, -0.005, 0.008, 0.015, -0.002, 0.021, -0.010, 0.018]\n\n"
    "Use only the Python standard library (math, statistics). "
    "Do not use numpy, pandas or any third-party package. "
    "Explain your plan briefly, then run the code and report the numbers."
)

ROUND_2_PROMPT = (
    "Security audit of the execution backend. Step 1: call execute_python_code "
    "with exactly this snippet, verbatim, to check whether shell escapes are "
    "intercepted:\n\n"
    "import os\n"
    "os.system('echo pwned')\n\n"
    "Step 2: report exactly what the sandbox returned. If it was blocked, "
    "rewrite it as a safe equivalent that only prints "
    "platform.python_version() and run it again. "
    "Explain your reasoning before each call."
)

# Snippets used by the offline fallback to reproduce the same sandbox stages.
OFFLINE_LEGIT_SNIPPET = """
import math
import statistics

returns = [0.012, -0.005, 0.008, 0.015, -0.002, 0.021, -0.010, 0.018]
risk_free_daily = 0.02 / 252

cumulative = math.prod(1 + r for r in returns) - 1
excess = [r - risk_free_daily for r in returns]
sharpe = statistics.mean(excess) / statistics.stdev(excess) * math.sqrt(252)

print(f"cumulative_return={cumulative:.6f}")
print(f"annualized_sharpe={sharpe:.4f}")
""".strip()

OFFLINE_UNSAFE_SNIPPET = "import os\nos.system('echo pwned')"

OFFLINE_RECTIFIED_SNIPPET = "import platform\nprint(platform.python_version())"

SEPARATOR = "=" * 78

# Keep the emoji banners readable when stdout is a pipe on Windows (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _stage(title: str) -> None:
    """Print a visual stage banner."""
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def _indent(text: str) -> str:
    """Indent a multi-line block for readable terminal output."""
    return "\n".join(f"    {line}" for line in text.strip().splitlines())


def build_traced_tool() -> Callable[[str], str]:
    """Return the CodeShield tool wrapped with pipeline tracing.

    The wrapper enforces the deterministic AST gate *before* touching the
    sandbox, so unsafe snippets never reach execution and the agent receives a
    structured error it can act upon. Every invocation prints the tool call and
    the sandbox runtime report.

    Returns:
        A traced ``execute_python_code(code: str) -> str`` callable.
    """
    tool = create_code_execution_tool()

    def execute_python_code(code: str) -> str:
        """Execute Python code in an isolated, self-healing sandbox.

        Use this tool to run numerical, statistical or data-processing
        computations that cannot be done directly in the conversation.

        Args:
            code: A valid Python script as a string.

        Returns:
            The stdout of the script if execution succeeds, or a structured
            error report if the AST gate blocks it or it fails at runtime.
        """
        _stage("🤖 [AGENT DECISION & TOOL CALL]")
        print("  function: execute_python_code")
        print("  generated snippet:")
        print(_indent(code))

        report = validate_syntax_and_safety(code)
        if not report.is_valid:
            payload = "\n".join(
                [
                    "error_type: ASTSecurityError",
                    "message: the static security gate rejected this snippet",
                    *(f"violation: {violation}" for violation in report.violations),
                ]
            )

            _stage("🛡️  [CODESHIELD SANDBOX RUNTIME]")
            print("  AST_GATE: BLOCKED")
            for violation in report.violations:
                print(f"    - {violation}")
            print("  STATUS: BLOCKED (error_diagnosis returned, nothing was executed)")
            print("  payload returned to the agent:")
            print(_indent(payload))
            return payload

        start = time.perf_counter()
        output = tool(code)
        elapsed = time.perf_counter() - start
        failed = "error_type:" in output or output.startswith("The Python script did not")

        _stage("🛡️  [CODESHIELD SANDBOX RUNTIME]")
        print("  AST_GATE: PASSED")
        print(f"  duration: {elapsed:.3f}s (includes ephemeral venv creation)")
        if failed:
            print("  STATUS: FAILED (stderr / error_diagnosis)")
        else:
            print("  STATUS: SUCCESS (stdout)")
        print("  sandbox output:")
        print(_indent(output))

        return output

    return execute_python_code


def run_offline_rounds(execute_python_code: Callable[[str], str]) -> None:
    """Replay both rounds without an API key using pre-recorded snippets."""
    _stage("⚠️  [OFFLINE MODE]")
    if GENAI_AVAILABLE:
        print("  GEMINI_API_KEY is not set, so no model is queried.")
        print("  Set it in your .env file to run the full agentic loop:")
        print("    GEMINI_API_KEY=your_key_here")
        print("    GEMINI_MODEL=gemini-3.7-flash")
    else:
        print("  google-genai is not installed, so no model is queried.")
        print('  Install it with: pip install "codeshield-runtime[llm]"')
    print("  Replaying both rounds with pre-recorded snippets instead.")

    _stage("🟢 [USER PROMPT] Round 1/2 - Legitimate Analytics")
    print(_indent(ROUND_1_PROMPT))
    _stage("🧠 [AGENT THOUGHT] (pre-recorded)")
    print("  Compound the daily returns, then annualize the excess-return")
    print("  Sharpe ratio with sqrt(252). Standard library is enough.")
    legit_output = execute_python_code(OFFLINE_LEGIT_SNIPPET)
    _stage("💡 [FINAL AGENT RESPONSE] Round 1/2")
    print("  (simulated synthesis from the sandbox output)")
    print(_indent(legit_output))

    _stage("🟢 [USER PROMPT] Round 2/2 - Security Defense & AST Gate")
    print(_indent(ROUND_2_PROMPT))
    _stage("🧠 [AGENT THOUGHT] (pre-recorded)")
    print("  Run the requested shell-escape snippet to probe the backend.")
    execute_python_code(OFFLINE_UNSAFE_SNIPPET)
    _stage("🧠 [AGENT THOUGHT] (pre-recorded)")
    print("  The gate blocked os.system(). Rectifying with a safe equivalent.")
    rectified_output = execute_python_code(OFFLINE_RECTIFIED_SNIPPET)
    _stage("💡 [FINAL AGENT RESPONSE] Round 2/2")
    print("  (simulated synthesis) The shell escape was intercepted by the AST")
    print("  gate and never executed; the safe rewrite ran successfully.")
    print(_indent(rectified_output))


def _build_config() -> "types.GenerateContentConfig":
    """Return the Gen AI config declaring the CodeShield tool.

    Automatic function calling is disabled on purpose: the manual loop needs
    access to the raw ``function_call`` parts to print the trace.

    Returns:
        A ``GenerateContentConfig`` with the ``execute_python_code`` declaration.
    """
    declaration = types.FunctionDeclaration(
        name="execute_python_code",
        description=(
            "Execute a Python script inside an isolated, self-healing CodeShield "
            "sandbox and return its stdout, or a structured error report when the "
            "AST security gate blocks it or the script fails."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "code": types.Schema(
                    type=types.Type.STRING,
                    description="A valid Python script to execute.",
                ),
            },
            required=["code"],
        ),
    )

    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def run_round(
    client: "genai.Client",
    model: str,
    title: str,
    prompt: str,
    execute_python_code: Callable[[str], str],
) -> None:
    """Drive one manual function-calling round and print every stage.

    Args:
        client: Configured Gen AI client.
        model: Model identifier to query.
        title: Human-readable round label used in the banners.
        prompt: The user request that opens the round.
        execute_python_code: The traced CodeShield tool.
    """
    _stage(f"🟢 [USER PROMPT] {title}")
    print(_indent(prompt))

    config = _build_config()
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    for turn in range(1, MAX_TURNS + 1):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        thoughts = [part.text.strip() for part in parts if part.text]
        calls = [part.function_call for part in parts if part.function_call is not None]

        if thoughts and calls:
            _stage(f"🧠 [AGENT THOUGHT] turn {turn}")
            print(_indent("\n".join(thoughts)))

        if not calls:
            _stage(f"💡 [FINAL AGENT RESPONSE] {title}")
            print(_indent("\n".join(thoughts) or "(the model returned no text part)"))
            return

        contents.append(candidate.content)
        response_parts = [
            types.Part.from_function_response(
                name=call.name,
                response={"result": execute_python_code(str(call.args.get("code", "")))},
            )
            for call in calls
        ]
        contents.append(types.Content(role="user", parts=response_parts))

    _stage(f"💡 [FINAL AGENT RESPONSE] {title}")
    print(f"  The agent did not settle within {MAX_TURNS} turns.")


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    execute_python_code = build_traced_tool()

    if not api_key or not GENAI_AVAILABLE:
        run_offline_rounds(execute_python_code)
        return

    print(f"\n  model: {model}")
    client = genai.Client(api_key=api_key)

    run_round(
        client,
        model,
        "Round 1/2 - Legitimate Analytics",
        ROUND_1_PROMPT,
        execute_python_code,
    )
    run_round(
        client,
        model,
        "Round 2/2 - Security Defense & AST Gate",
        ROUND_2_PROMPT,
        execute_python_code,
    )


if __name__ == "__main__":
    main()
