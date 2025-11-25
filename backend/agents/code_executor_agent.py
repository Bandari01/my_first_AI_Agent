import re
from pathlib import Path
from typing import List

from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient
from backend.kaggle.data_fetcher import CompetitionInfo
from .protocols import Plan, PlanStep, ExecutionArtifact
from .tool_registry import ToolRegistry


logger = get_logger(__name__)


class CodeExecutorAgent:
    """Tool-calling agent that implements plan steps via tools.

    For 'execute_python' steps, it uses the LLM to synthesize a complete script and
    runs it via the ToolRegistry. For 'validate_submission', it calls the registry
    validator. It tracks created files and execution outputs.
    """

    def __init__(self, working_dir: Path, tool_registry: ToolRegistry, llm: LLMClient | None = None):
        self.working_dir = Path(working_dir)
        self.tool_registry = tool_registry
        self.llm = llm or LLMClient()

    def execute_plan(self, plan: Plan, info: CompetitionInfo, data_summary: str) -> List[ExecutionArtifact]:
        artifacts: List[ExecutionArtifact] = []
        for step in plan.steps:
            if not step.tool_call:
                logger.warning(f"Step {step.id} has no tool_call; marking as failed")
                artifacts.append(ExecutionArtifact(step_id=step.id, success=False, error_message="missing tool_call"))
                continue

            name = step.tool_call.name
            if name == "execute_python":
                code = self._generate_code_for_step(step, info, data_summary)
                pre_files = {p.name for p in self.working_dir.glob('*')}
                result = self.tool_registry.call("execute_python", {"code": code})
                post_files = {p.name for p in self.working_dir.glob('*')}
                created = sorted(list(post_files - pre_files))
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.id,
                        success=result.success,
                        stdout=result.output or "",
                        stderr=result.error,
                        files_created=created,
                        duration_seconds=float(result.artifacts.get("execution_time", 0.0)),
                        error_message=result.error,
                    )
                )
            elif name == "validate_submission":
                path = step.tool_call.params.get("path", str(self.working_dir / "submission.csv"))
                result = self.tool_registry.call("validate_submission", {"path": path})
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.id,
                        success=result.success,
                        stdout=result.output or "",
                        stderr=result.error,
                        files_created=[],
                        duration_seconds=0.0,
                        error_message=result.error,
                        code=code,
                    )
                )
            else:
                logger.warning(f"Unknown tool {name} for step {step.id}")
                artifacts.append(ExecutionArtifact(step_id=step.id, success=False, error_message=f"unknown tool: {name}"))

        return artifacts

    def _generate_code_for_step(self, step: PlanStep, info: CompetitionInfo, data_summary: str) -> str:
        """Ask the LLM to synthesize a complete Python script for this step.

        The script runs in a fresh Python process, must be fully self-contained,
        operate only on local CSV files in the working directory, and, when applicable,
        save predictions to 'submission.csv' following the sample submission columns.
        """
        train_files = ", ".join(info.train_files) if info.train_files else "(none)"
        test_files = ", ".join(info.test_files) if info.test_files else "(none)"
        sample_submission = info.sample_submission_file or "(none)"
        working_dir_str = str(self.working_dir)

        constraints = (
            "Constraints:\n"
            "- This script runs in a FRESH Python process. It CANNOT see variables from other steps.\n"
            "- You MUST import all required libraries inside this script (e.g. import pandas as pd).\n"
            f"- The competition working directory is: {working_dir_str}\n"
            "- Load all CSV files from this directory using pandas.read_csv and REAL file names.\n"
            f"- VALID filenames are ONLY: {train_files}, {test_files}, {sample_submission}.\n"
            "- NEVER use fake names like 'your_data.csv' or 'path_to_your_data.csv'.\n"
            "- Do NOT assume variables like 'train', 'test', 'merged_train', 'merged_test', or 'model' already exist; you MUST create them in THIS script.\n"
            "- Do NOT use placeholders like '...' or ellipsis objects. All code must be fully implemented and executable.\n"
            "- Use only local files in the working directory (no downloads, no Kaggle API calls).\n"
            "- Preferred libs: pandas, numpy, scikit-learn; keep the code simple and robust.\n"
            "- Avoid internet access and external APIs.\n"
            "- Print a few concise progress logs so a human can follow what the script is doing.\n"
            "- When creating a submission file:\n"
            "  * Load the FULL sample submission CSV.\n"
            "  * Use its 'id' column EXACTLY as-is (no filtering, no sampling, no .head()).\n"
            "  * Generate ONE prediction per row in sample_submission.\n"
            "  * Replace ONLY the target column (e.g. 'sales') with your predictions.\n"
            f"  * Save the final result to '{working_dir_str}/submission.csv' with the SAME columns and row count as the sample.\n"
        )

        prompt = (
            "You are writing a COMPLETE, STANDALONE Python script for a Kaggle competition step.\n"
            "The script will be executed in isolation, in a fresh Python process.\n"
            "It must not rely on any global state or variables from previous steps.\n\n"
            f"Step title: {step.title}\n"
            f"Step objective: {step.objective}\n"
            f"Competition id: {info.competition_id}\n"
            f"Problem type: {info.problem_type}\n"
            f"Metric: {info.evaluation_metric}\n"
            f"Training files: {train_files}\n"
            f"Test files: {test_files}\n"
            f"Sample submission: {sample_submission}\n\n"
            "Data summary:\n" + data_summary + "\n\n" +
            constraints +
            "\nReturn ONLY code (prefer a ```python fenced block)."
        )

        resp = self.llm.generate(prompt)
        return self._extract_code(resp.content)

    def _extract_code(self, text: str) -> str:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()