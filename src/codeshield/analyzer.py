"""Static syntax and safety analysis using only the standard ``ast`` module."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from codeshield.schemas import ValidationReport

logger = logging.getLogger(__name__)


# Parametrizable functions considered dangerous when invoked directly.
_DANGEROUS_NAMES: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "os.system",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
})
_GENERIC_EXCEPTIONS: frozenset[str] = frozenset({"Exception", "BaseException"})


@dataclass
class _SafetyContext:
    """Mutable collector used by the AST visitor."""

    violations: list[str] = field(default_factory=list)
    line_numbers: dict[int, str] = field(default_factory=dict)

    def add_violation(self, message: str, *, line: int | None = None) -> None:
        """Append a violation with an optional source line hint."""
        if line is not None:
            message = f"Line {line}: {message}"
        self.violations.append(message)


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that flags generic exception handlers and dangerous calls."""

    def __init__(self, context: _SafetyContext) -> None:
        self._context = context

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        """Detect overly broad exception handlers."""
        self._check_generic_except(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Detect dangerous parametrizable function calls."""
        self._check_dangerous_call(node)
        self.generic_visit(node)

    def _check_generic_except(self, node: ast.ExceptHandler) -> None:
        """Flag ``except:``, ``except Exception:`` and ``except BaseException:``."""
        if node.type is None:
            self._context.add_violation(
                "Bare ``except:`` clause is not allowed because it catches all exceptions",
                line=node.lineno,
            )
            return

        name = self._name_from_node(node.type)
        if name in _GENERIC_EXCEPTIONS:
            self._context.add_violation(
                f"Overly generic exception handler ``except {name}:`` is not allowed",
                line=node.lineno,
            )

    def _check_dangerous_call(self, node: ast.Call) -> None:
        """Flag direct calls to eval/exec/compile."""
        name = self._name_from_node(node.func)
        if name in _DANGEROUS_NAMES:
            self._context.add_violation(
                f"Dangerous parametrizable function call ``{name}()`` is not allowed",
                line=node.lineno,
            )

    @staticmethod
    def _name_from_node(node: ast.AST | None) -> str:
        """Return a best-effort dotted name for an AST node."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_SafetyVisitor._name_from_node(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return _SafetyVisitor._name_from_node(node.value)
        return ""


def _lines_from_source(code: str) -> dict[int, str]:
    """Map 1-indexed line numbers to stripped source lines."""
    return {idx + 1: line for idx, line in enumerate(code.splitlines())}


def _split_first_line(code: str) -> tuple[str, ...]:
    """Return source lines as a tuple to keep them hashable when needed."""
    return tuple(code.splitlines())


def validate_syntax_and_safety(code: str) -> ValidationReport:
    """Validate ``code`` using the standard ``ast`` module.

    The validation detects:
    - Syntax errors raised by ``ast.parse``.
    - Bare ``except:`` clauses and handlers using ``Exception``/``BaseException``.
    - Direct calls to dangerous parametrizable functions: ``eval``, ``exec`` and ``compile``.

    Args:
        code: Python source code to validate.

    Returns:
        A ``ValidationReport`` containing the result and any violations.
    """
    lines = _lines_from_source(code)
    try:
        tree = ast.parse(code, filename="<dynamic>", mode="exec")
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown location"
        text = lines.get(exc.lineno, "") if exc.lineno else ""
        pointer = f"\n{text}\n{' ' * ((exc.offset or 1) - 1)}^" if exc.offset and text else ""
        message = f"SyntaxError at {location}: {exc.msg}{pointer}"
        logger.debug("Syntax validation failed: %s", message)
        return ValidationReport(
            is_valid=False,
            violations=[message],
            exception=exc,
        )

    context = _SafetyContext(line_numbers=lines)
    visitor = _SafetyVisitor(context)
    visitor.visit(tree)

    is_valid = not context.violations
    logger.debug(
        "Safety validation completed: %d violation(s) found.",
        len(context.violations),
    )
    return ValidationReport(
        is_valid=is_valid,
        violations=context.violations,
        exception=None,
    )


__all__: list[str] = ["validate_syntax_and_safety"]
