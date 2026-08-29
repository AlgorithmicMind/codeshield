"""Example 4: real-world agentic function calling with CodeShield + Gemini."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from codeshield import create_code_execution_tool


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    if not api_key:
        print("GEMINI_API_KEY not set. Set it in .env to run this example.")
        return

    execute_python_code = create_code_execution_tool()

    client = genai.Client(api_key=api_key)

    prompt = (
        "Calculate the cumulative return and the annualized Sharpe Ratio "
        "(assuming a risk-free rate of 0.02) for this daily return series:\n"
        "[0.012, -0.005, 0.008, 0.015, -0.002, 0.021, -0.010, 0.018]\n\n"
        "Use only the Python standard library (math, statistics). "
        "Do not use numpy, pandas or any third-party package. "
        "Print the cumulative return and the annualized Sharpe ratio clearly."
    )

    # Chats resolve the full automatic function-calling loop and return text.
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            tools=[execute_python_code],
            temperature=0.2,
        ),
    )

    response = chat.send_message(prompt)

    print("\n--- Final response ---")
    print(response.text)


if __name__ == "__main__":
    main()
