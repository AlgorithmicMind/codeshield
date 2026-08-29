"""CodeShield.

A secure, isolated, and self-healing Python code execution engine.
"""

__version__ = "0.1.0"

from codeshield.loop import SelfHealingEngine
from codeshield.schemas import CodeExecutionRequest, ExecutionResult

__all__ = ["SelfHealingEngine", "CodeExecutionRequest", "ExecutionResult"]
