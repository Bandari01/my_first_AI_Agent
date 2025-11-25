"""
Simplified code executor

Features:
- Execute generated Python code in an isolated subprocess
- If execution fails (non-zero return code or exception), send the traceback and original code to the project's LLM client to request a fixed version
- Retry execution with the fixed code up to a maximum number of attempts

Note: This module depends on `backend.llm.llm_client.LLMClient.generate(prompt)`. The return object's `.content` should contain the fixed code.
"""
from dataclasses import dataclass
import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path
from typing import Optional
import re


from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    return_code: int = 0


class GeneratedCodeExecutor:
    """Execute generated code and request LLM-based fixes on failure."""

    def __init__(self, timeout: int = 300, max_fix_attempts: int = 1):
        self.timeout = timeout
        self.max_fix_attempts = max_fix_attempts
        # Initialize LLM client (uses default environment configuration)
        try:
            self.llm = LLMClient()
        except Exception as e:
            logger.warning(f"Cannot initialize LLMClient: {e}")
            self.llm = None

    def _strip_markdown_code_fences(self, code: str) -> str:
        """
        Remove Markdown code fences like ```python ... ``` if present.
        """
        if not code:
            return ""

        code = code.strip()

        if code.startswith("```"):
            lines = code.splitlines()
            if lines:
                first = lines[0].strip()
                if first.startswith("```"):
                    lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        return code

    def _run_subprocess(self, code: str, working_dir: Optional[Path] = None, env: Optional[dict] = None) -> ExecutionResult:
        start = time.time()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            tmp_path = Path(f.name)
            f.write(code)

        cwd = str(working_dir) if working_dir else None
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)

        try:
            proc = subprocess.Popen([sys.executable, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env_vars, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return ExecutionResult(False, stdout, error=f"Timeout after {self.timeout}s", execution_time=time.time()-start, return_code=-1)

            return ExecutionResult(proc.returncode == 0, stdout, error=stderr if stderr else None, execution_time=time.time()-start, return_code=proc.returncode)
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _ask_llm_to_fix(self, original_code: str, traceback_text: str) -> Optional[str]:
        if not self.llm:
            logger.error("LLM client not initialized; cannot request fix")
            return None

        prompt = (
            "Your task: Fix the following Python code.\n"
            "Requirement: Return ONLY the corrected, full Python code.\n\n"
            "=== Original Code ===\n"
            f"{original_code}\n\n"
            "=== Error / Traceback ===\n"
            f"{traceback_text}\n\n"
            "Return only the fixed code, with no explanations or extra text."
        )

        try:
            resp = self.llm.generate(prompt)
            raw = (resp.content if hasattr(resp, "content") else str(resp))
            fixed = self._strip_markdown_code_fences(raw)
            return fixed
        except Exception as e:
            logger.error(f"Request to LLM for fix failed: {e}")
            return None

    def execute_and_autofix(self, code: str, working_dir: Optional[Path] = None, env: Optional[dict] = None) -> ExecutionResult:
        """Execute code; on failure, request LLM fix and retry up to max attempts."""
        attempt = 0
        current_code = self._strip_markdown_code_fences(code)

        while True:
            attempt += 1
            logger.info(f"Execute code, attempt #{attempt}")
            result = self._run_subprocess(current_code, working_dir, env)

            if result.success:
                logger.info(f"Code executed successfully (attempt={attempt}, time={result.execution_time:.2f}s)")
                return result

            # Execution failed
            tb = result.error or ""
            error_summary = tb.strip()
            error_message = f"Execution failed (return_code={result.return_code})" + (f"\nError: {error_summary[:400]}..." if error_summary else "")
            logger.warning(f"{error_message}. Preparing to request LLM fix")

            if attempt > self.max_fix_attempts:
                logger.error(f"Reached maximum fix attempts ({self.max_fix_attempts}); stopping retries")
                return result

            fixed = self._ask_llm_to_fix(current_code, tb or "No stderr captured")
            if fixed:
                logger.info("LLM returned fixed code, preparing next attempt...")
            if not fixed:
                logger.error("LLM did not return fixed code or request failed; stopping retries")
                return result

            logger.info("LLM returned fixed code; proceeding to next attempt")
            current_code = fixed


