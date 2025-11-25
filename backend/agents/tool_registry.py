import json
from pathlib import Path
from typing import Dict, Any, Optional

from backend.config import config
from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient
from backend.executor.code_executor import GeneratedCodeExecutor
from backend.kaggle.submission_validator import SubmissionValidator
from .protocols import ToolResult


logger = get_logger(__name__)


class ToolRegistry:
    """Registry that maps tool names to concrete implementations.

    Supported tools:
    - execute_python: runs Python code via a subprocess and auto-fixes errors using LLM
    - validate_submission: validates 'submission.csv' against sample submission if available
    """

    def __init__(
        self,
        working_dir: Path,
        llm_client: Optional[LLMClient] = None,
        sample_submission_path: Optional[Path] = None,
        timeout: Optional[int] = None,
        max_fix_attempts: Optional[int] = None,
    ):
        self.working_dir = Path(working_dir)
        self.llm_client = llm_client or LLMClient()
        self.sample_submission_path = sample_submission_path
        self.timeout = timeout or config.max_execution_time
        self.max_fix_attempts = max_fix_attempts or 1

    def call(self, name: str, params: Dict[str, Any]) -> ToolResult:
        if name == "execute_python":
            return self._execute_python(params)
        if name == "validate_submission":
            return self._validate_submission(params)
        return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _execute_python(self, params: Dict[str, Any]) -> ToolResult:
        code: str = params.get("code", "")
        if not code:
            return ToolResult(success=False, error="'code' parameter is required for execute_python")

        executor = GeneratedCodeExecutor(timeout=self.timeout, max_fix_attempts=self.max_fix_attempts)
        result = executor.execute_and_autofix(code, working_dir=self.working_dir)

        artifacts = {
            "execution_time": result.execution_time,
            "return_code": result.return_code,
        }
        return ToolResult(success=result.success, output=result.output, error=result.error, artifacts=artifacts)

    def _validate_submission(self, params: Dict[str, Any]) -> ToolResult:
        path_param = params.get("path")
        submission_path = Path(path_param) if path_param else (self.working_dir / "submission.csv")

        validator = SubmissionValidator(
            sample_submission_path=self.sample_submission_path if (self.sample_submission_path and self.sample_submission_path.exists()) else None
        )

        is_valid, errors = validator.validate(submission_path)
        summary = validator.get_submission_summary(submission_path)

        output = json.dumps(summary, ensure_ascii=False, indent=2)
        error_text = "\n".join(errors) if errors else None

        return ToolResult(success=is_valid, output=output, error=error_text, artifacts={"summary": summary, "errors": errors})