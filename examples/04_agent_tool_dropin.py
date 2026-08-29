"""Example 4: real-world agentic function calling with CodeShield + Gemini.

The integration itself is only three statements -- everything else in this file
exists to *print* what normally stays invisible::

    execute_python_code = create_code_execution_tool()
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(tools=[execute_python_code]),
    )
    print(chat.send_message(prompt).text)

Automatic function calling (the SDK default for plain Python callables) handles
the call/response cycle, so no manual protocol loop is needed.

Two rounds are played:

1. **Legitimate Analytics**: the agent reasons, writes a financial computation,
   clears the AST gate and executes it inside the ephemeral uv sandbox.
2. **Security Defense**: the agent tries a shell escape, the AST gate blocks it
   before execution, and the structured error is fed back so it rectifies.

Each round prints, in chronological order: ``[USER PROMPT]``,
``[AGENT THOUGHT]``, ``[AGENT DECISION & TOOL CALL]``,
``[CODESHIELD SANDBOX RUNTIME]`` and ``[FINAL AGENT RESPONSE]``.

Without ``GEMINI_API_KEY`` both rounds are replayed offline against the very
same sandbox with pre-recorded snippets, so the script never fails in CI.
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


def build_traced_tool(reports: list[str]) -> Callable[[str], str]:
    """Return the CodeShield tool, recording one sandbox report per call.

    The AST gate runs *before* the sandbox, so unsafe snippets never reach
    execution and the agent receives a structured error it can act upon.
    Reports are buffered instead of printed so the trace can be replayed in
    chronological order next to the model's reasoning.

    Args:
        reports: Sink that receives one rendered report per tool call.

    Returns:
        An ``execute_python_code(code: str) -> str`` callable for the SDK.
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
        validation = validate_syntax_and_safety(code)

        if not validation.is_valid:
            payload = "\n".join(
                [
                    "error_type: ASTSecurityError",
                    "message: the static security gate rejected this snippet",
                    *(f"violation: {item}" for item in validation.violations),
                ]
            )
            reports.append(
                "\n".join(
                    [
                        "  AST_GATE: BLOCKED",
                        *(f"    - {item}" for item in validation.violations),
                        "  STATUS: BLOCKED (nothing was executed)",
                        "  payload returned to the agent:",
                        _indent(payload),
                    ]
                )
            )
            return payload

        start = time.perf_counter()
        output = tool(code)
        elapsed = time.perf_counter() - start
        failed = "error_type:" in output
        status = "FAILED (stderr / error_diagnosis)" if failed else "SUCCESS (stdout)"

        reports.append(
            "\n".join(
                [
                    "  AST_GATE: PASSED",
                    f"  duration: {elapsed:.3f}s (includes ephemeral venv creation)",
                    f"  STATUS: {status}",
                    "  sandbox output:",
                    _indent(output),
                ]
            )
        )
        return output

    return execute_python_code


def _print_tool_call(code: str, report: str) -> None:
    """Print the tool call and its buffered sandbox report."""
    _stage("🤖 [AGENT DECISION & TOOL CALL]")
    print("  function: execute_python_code")
    print("  generated snippet:")
    print(_indent(code))

    _stage("🛡️  [CODESHIELD SANDBOX RUNTIME]")
    print(report)


def print_trace(history: list, reports: list[str]) -> None:
    """Replay one round chronologically from the chat history.

    Automatic function calling already resolved the cycle, so the history holds
    the model's reasoning text and its ``function_call`` parts in order; the
    matching sandbox reports are popped from ``reports``.

    Args:
        history: Entries returned by ``chat.get_history()``.
        reports: Sandbox reports recorded by the traced tool, in call order.
    """
    pending = list(reports)
    turns = [entry for entry in history if entry.role == "model"]

    for entry in turns[:-1]:  # the last model turn is the final answer
        for part in entry.parts or []:
            if part.text and part.text.strip():
                _stage("🧠 [AGENT THOUGHT]")
                print(_indent(part.text))
            elif part.function_call is not None and pending:
                code = str(part.function_call.args.get("code", ""))
                _print_tool_call(code, pending.pop(0))


def run_round(client: "genai.Client", model: str, title: str, prompt: str) -> None:
    """Run one round with automatic function calling and print its trace.

    Args:
        client: Configured Gen AI client.
        model: Model identifier to query.
        title: Round label used in the banners.
        prompt: The user request that opens the round.
    """
    reports: list[str] = []
    execute_python_code = build_traced_tool(reports)

    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            tools=[execute_python_code],
            temperature=0.2,
        ),
    )

    _stage(f"🟢 [USER PROMPT] {title}")
    print(_indent(prompt))

    response = chat.send_message(prompt)

    print_trace(chat.get_history(), reports)

    _stage(f"💡 [FINAL AGENT RESPONSE] {title}")
    print(_indent(response.text or "(the model returned no text part)"))


def run_offline_rounds() -> None:
    """Replay both rounds without an API key using pre-recorded snippets."""
    reports: list[str] = []
    execute_python_code = build_traced_tool(reports)

    def replay(thought: str, snippet: str) -> str:
        """Run one pre-recorded tool call and print its stages."""
        _stage("🧠 [AGENT THOUGHT] (pre-recorded)")
        print(_indent(thought))
        output = execute_python_code(snippet)
        _print_tool_call(snippet, reports.pop())
        return output

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
    analytics = replay(
        "Compound the daily returns, then annualize the excess-return Sharpe\n"
        "ratio with sqrt(252). The standard library is enough.",
        OFFLINE_LEGIT_SNIPPET,
    )
    _stage("💡 [FINAL AGENT RESPONSE] Round 1/2 - Legitimate Analytics")
    print(_indent(analytics))

    _stage("🟢 [USER PROMPT] Round 2/2 - Security Defense & AST Gate")
    print(_indent(ROUND_2_PROMPT))
    replay("Probe the backend with the requested shell escape.", OFFLINE_UNSAFE_SNIPPET)
    rectified = replay(
        "The gate blocked os.system() and nothing ran. Rectifying with a safe\nequivalent.",
        OFFLINE_RECTIFIED_SNIPPET,
    )
    _stage("💡 [FINAL AGENT RESPONSE] Round 2/2 - Security Defense & AST Gate")
    print("  The shell escape was intercepted before execution; the safe")
    print("  rewrite then ran successfully and returned:")
    print(_indent(rectified))


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

    if not api_key or not GENAI_AVAILABLE:
        run_offline_rounds()
        return

    print(f"\n  model: {model}")
    client = genai.Client(api_key=api_key)

    run_round(client, model, "Round 1/2 - Legitimate Analytics", ROUND_1_PROMPT)
    run_round(client, model, "Round 2/2 - Security Defense & AST Gate", ROUND_2_PROMPT)


if __name__ == "__main__":
    main()
