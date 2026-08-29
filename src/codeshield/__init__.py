"""CodeShield.

A secure, isolated, and self-healing Python code execution engine.
"""

__version__ = "0.1.2"

from codeshield.analyzer import validate_syntax_and_safety
from codeshield.classifier import TracebackClassifier
from codeshield.environment import SandboxError, SandboxManager
from codeshield.loop import (
    ASTSecurityError,
    GeminiPatchGenerator,
    LLMPatchError,
    SelfHealingEngine,
    SelfHealingError,
)
from codeshield.runner import SubprocessRunner, SubprocessRunnerError
from codeshield.schemas import (
    CodeExecutionRequest,
    ErrorDiagnosis,
    ExecutionResult,
    PatchProposal,
    ValidationReport,
)
from codeshield.tools import create_code_execution_tool

__all__ = [
    "ASTSecurityError",
    "CodeExecutionRequest",
    "ErrorDiagnosis",
    "ExecutionResult",
    "GeminiPatchGenerator",
    "LLMPatchError",
    "PatchProposal",
    "SandboxError",
    "SandboxManager",
    "SelfHealingEngine",
    "SelfHealingError",
    "SubprocessRunner",
    "SubprocessRunnerError",
    "TracebackClassifier",
    "ValidationReport",
    "create_code_execution_tool",
    "validate_syntax_and_safety",
]
