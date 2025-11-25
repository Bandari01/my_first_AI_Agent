import time
from pathlib import Path
from typing import Optional

from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient
from backend.kaggle.data_fetcher import KaggleDataFetcher, CompetitionInfo
from backend.config import config
from .protocols import OrchestratorResult, Plan, ExecutionArtifact
from .multi_agent import DataAnalystAgent
from .tool_registry import ToolRegistry
from .code_executor_agent import CodeExecutorAgent
from typing import List, Optional
from .protocols import ExecutionArtifact


logger = get_logger(__name__)


class MultiAgentOrchestrator:
    """Coordinates the CAG planning agent and the tool-calling executor.

    Flow:
    1) Fetch Kaggle competition info and data summary
    2) DataAnalystAgent produces a hierarchical plan
    3) CodeExecutorAgent executes the plan via tool calls
    4) Optionally validate the submission
    """

    def __init__(self, llm: Optional[LLMClient] = None, max_fix_attempts: int = 1, max_execution_time: Optional[int] = None):
        self.llm = llm or LLMClient()
        self.fetcher = KaggleDataFetcher()
        self.max_fix_attempts = max_fix_attempts
        self.max_execution_time = max_execution_time or config.max_execution_time

    def run(self, competition_url: str, validate: bool = True, force_download: bool = False) -> OrchestratorResult:
        start = time.time()
        errors = []
        llm_calls = 0

        try:
            info: CompetitionInfo = self.fetcher.fetch_complete_info(competition_url, download_data=True, force_download=force_download)
        except Exception as e:
            logger.error(f"Failed to fetch competition info: {e}")
            return OrchestratorResult(
                success=False,
                plan=Plan(goal="Failed to prepare"),
                artifacts=[],
                errors=[str(e)],
                total_time_seconds=time.time() - start,
            )

        problem_description = info.description or f"Solve Kaggle competition: {info.competition_name}"
        data_summary = self.fetcher.get_data_summary(info)
        working_dir = info.data_path or (config.competitions_dir / info.competition_id)

        # 1) Plan
        planner = DataAnalystAgent(llm=self.llm)
        plan = planner.plan(problem_description, info, data_summary)
        llm_calls += 1

        # Tool registry and executor
        sample_submission_path = None
        if info.sample_submission_file and working_dir:
            candidate = Path(working_dir) / info.sample_submission_file
            if candidate.exists():
                sample_submission_path = candidate

        registry = ToolRegistry(
            working_dir=Path(working_dir),
            llm_client=self.llm,
            sample_submission_path=sample_submission_path,
            timeout=self.max_execution_time,
            max_fix_attempts=self.max_fix_attempts,
        )
        executor = CodeExecutorAgent(working_dir=Path(working_dir), tool_registry=registry, llm=self.llm)

        artifacts: list[ExecutionArtifact] = executor.execute_plan(plan, info, data_summary)
        print("\n===== DEBUG: artifacts list =====")
        print(artifacts)
        generated_code = self._collect_generated_code(artifacts)
        llm_calls += sum(1 for a in artifacts if a.step_id and a.step_id != "" and a.stdout is not None)  # rough signal of LLM code generation calls

        # Locate submission
        submission_path = Path(working_dir) / "submission.csv"
        if not submission_path.exists():
            submission_path = None

        validation = None
        if validate and submission_path is not None:
            result = registry.call("validate_submission", {"path": str(submission_path)})
            validation = {"success": result.success, "output": result.output, "error": result.error}

        if validation is not None:
            success = bool(validation.get("success", False))
        else:
            success = any(a.success for a in artifacts)
        total_time = time.time() - start

        # ---- Normalize booleans to avoid numpy.bool_ JSON issue ----

        # ensure success is pure Python bool
        success = bool(success)

        # sanitize validation dict if exists
        if validation:
            validation = {
                k: bool(v) if hasattr(v, "item") or isinstance(v, (bool,)) else v
                for k, v in validation.items()
            }

        return OrchestratorResult(
            success=success,
            plan=plan,
            artifacts=artifacts,
            submission_path=submission_path,
            validation=validation,
            errors=errors,
            llm_calls=llm_calls,
            total_time_seconds=total_time,
            generated_code=generated_code,
        )
 
    def _collect_generated_code(self, artifacts: list["ExecutionArtifact"]) -> Optional[str]:
        """
        Try to collect any generated Python code from ExecutionArtifacts.

        Priority:
        1) If an artifact has `code` (the Python source), use that.
        2) Else, if artifacts record created .py files in `files_created`, read those files.
        3) If nothing else is available, fall back to concatenating stdout from each step.
        """
        code_chunks: list[str] = []

        for art in artifacts:
            code_text = getattr(art, "code", None)
            if code_text:
                header = f"# ===== Step {getattr(art, 'step_id', '')} | generated code =====\n"
                code_chunks.append(header + code_text)

        if not code_chunks:
            for art in artifacts:
                files = getattr(art, "files_created", []) or []
                for fpath in files:
                    if isinstance(fpath, str) and fpath.endswith(".py"):
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                file_code = f.read()
                            header = f"# ===== Step {getattr(art, 'step_id', '')} | generated file: {fpath} =====\n"
                            code_chunks.append(header + file_code)
                        except OSError:
                            continue

        if not code_chunks:
            for art in artifacts:
                stdout = getattr(art, "stdout", None)
                if stdout:
                    header = f"# ===== Step {getattr(art, 'step_id', '')} | stdout log =====\n"
                    code_chunks.append(header + stdout)

        if not code_chunks:
            return None

        return "\n\n".join(code_chunks)
