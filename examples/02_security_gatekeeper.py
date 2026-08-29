"""Example 2: demonstrate AST security gatekeeper rejecting unsafe code."""

from __future__ import annotations

from codeshield import validate_syntax_and_safety


def main() -> None:
    samples = [
        ("Syntax error", 'print("hello"'),
        ("Bare except", "try:\n    x = 1\nexcept:\n    pass"),
        ("Dangerous eval", "result = eval('1 + 1')"),
        ("OS system call", "import os; os.system('echo pwned')"),
        ("Subprocess shell", "import subprocess; subprocess.run('ls', shell=True)"),
    ]

    for title, code in samples:
        print(f"--- {title} ---")
        report = validate_syntax_and_safety(code)
        if report.is_valid:
            print("  OK: code passed AST validation")
        else:
            print("  BLOCKED:")
            for violation in report.violations:
                print(f"    - {violation}")


if __name__ == "__main__":
    main()
