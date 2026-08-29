"""Pydantic models for the execution engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeExecutionRequest(BaseModel):
    """Request to execute a piece of Python code safely."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1, description="Python source code to execute.")
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=3600.0,
        description="Maximum execution time in seconds.",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Optional list of PyPI packages to install in the sandbox.",
    )
    file_name: str | None = Field(
        default=None,
        description="Optional file name used when persisting code in the sandbox.",
    )

    @field_validator("code")
    @classmethod
    def _code_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Code must contain non-whitespace characters.")
        return value


class ExecutionResult(BaseModel):
    """Result of an isolated code execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stdout: str = Field(default="", description="Captured standard output.")
    stderr: str = Field(default="", description="Captured standard error.")
    exit_code: int | None = Field(
        default=None,
        description="Process exit code; None if the process did not terminate normally.",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock execution time in seconds.",
    )
    silent_failure_detected: bool = Field(
        default=False,
        description="True when exit_code is 0 but suspicious patterns were detected in output.",
    )
    timed_out: bool = Field(
        default=False,
        description="True when the process was terminated due to timeout.",
    )


class ErrorDiagnosis(BaseModel):
    """Structured diagnosis extracted from a runtime traceback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str = Field(..., description="Exception class name, e.g. NameError.")
    root_cause_line: int | None = Field(
        default=None,
        description="Line number in the original source where the failure originated.",
    )
    message: str = Field(default="", description="Exception message or human-readable summary.")
    context: list[str] = Field(
        default_factory=list,
        description="Surrounding source lines for the failing line, if available.",
    )


class PatchProposal(BaseModel):
    """A validated patch proposal ready to be applied to a source file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: Path = Field(..., description="Target source file in the sandbox.")
    patched_code: str = Field(..., description="Proposed replacement source code.")
    is_syntax_valid: bool = Field(
        default=False,
        description="True when the patched code passed AST validation.",
    )
    diagnosis: ErrorDiagnosis | None = Field(
        default=None,
        description="Diagnosis that motivated the patch.",
    )


class ValidationReport(BaseModel):
    """Report produced by static AST validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    is_valid: bool = Field(..., description="True when no syntax or safety violations were found.")
    violations: list[str] = Field(
        default_factory=list,
        description="Human-readable list of syntax and safety violations.",
    )
    exception: SyntaxError | None = Field(
        default=None,
        description="Original SyntaxError instance, if any.",
    )

    def model_post_init(self, __context: Any) -> None:  # noqa: N807
        """Ensure is_valid remains consistent with the violations list."""
        if self.is_valid and self.violations:
            object.__setattr__(self, "is_valid", False)
        elif not self.is_valid and not self.violations and self.exception is None:
            object.__setattr__(self, "is_valid", True)
