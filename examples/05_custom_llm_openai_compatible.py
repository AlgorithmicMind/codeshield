"""Example 5: self-healing with a custom frontier-model patch generator.

This example simulates a frontier model (OpenAI, Anthropic, DeepSeek, Ollama)
without requiring any API key. In production, replace ``mock_frontier_patcher``
with a real call to any of the providers shown in the comments.
"""

from __future__ import annotations

from codeshield import SelfHealingEngine


def mock_frontier_patcher(code: str, diagnosis) -> str | None:
    """Return a corrected snippet based on the runtime diagnosis.

    Replace this function body with a real frontier-model call. Supported
    model identifiers include:

    - ``gpt-5.6-luna`` (OpenAI)
    - ``claude-sonnet-5`` (Anthropic)
    - ``deepseek-v4-flash`` (DeepSeek)
    - Any Ollama local model (e.g. ``llama3.2``)

    OpenAI example:

        import openai

        response = openai.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Fix this Python code:\n{code}\n"
                        f"Error: {diagnosis.error_type}: {diagnosis.message}"
                    ),
                }
            ],
        )
        return response.choices[0].message.content

    Anthropic example:

        import anthropic

        response = anthropic.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Fix this Python code:\n{code}\n"
                        f"Error: {diagnosis.error_type}: {diagnosis.message}"
                    ),
                }
            ],
        )
        return response.content[0].text
    """
    if diagnosis.error_type == "TypeError":
        # Fix: cannot concatenate str and int; convert the number to a string.
        return code.replace('"Result: " + 42', '"Result: " + str(42)')

    if diagnosis.error_type == "IndexError":
        # Fix: out-of-bounds access; clamp to the first element.
        return code.replace("[10]", "[0]")

    return None


def main() -> None:
    code = 'print("Result: " + 42)'

    print("1) Original broken code:")
    print(f"   {code!r}")
    print("\n2) Running with a mock frontier-model patch generator...")

    engine = SelfHealingEngine(
        patch_generator=mock_frontier_patcher,
        use_llm=False,
    )
    with engine:
        result, diagnosis = engine.run(code)

    print("\n3) Repaired execution:")
    print(f"   Exit code: {result.exit_code}")
    print(f"   Stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"   Stderr: {result.stderr.strip()}")
    if diagnosis:
        print(f"   Diagnosis: {diagnosis.error_type} - {diagnosis.message}")


if __name__ == "__main__":
    main()
