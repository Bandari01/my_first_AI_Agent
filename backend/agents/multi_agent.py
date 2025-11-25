import json
import re
from typing import Optional, List, Dict, Any

from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient
from backend.kaggle.data_fetcher import CompetitionInfo
from .protocols import Plan, PlanStep, ToolCall


logger = get_logger(__name__)


class DataAnalystAgent:
    """CAG-style planning agent that produces a hierarchical execution plan.

    The agent focuses on strategy and decomposes the problem into tool-callable steps.
    It does NOT implement code itself; the CodeExecutorAgent handles implementation.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def plan(self, problem_description: str, info: CompetitionInfo, data_summary: str) -> Plan:
        """Generate a plan using the LLM, returning a structured Plan dataclass.

        The plan consists of step-by-step tool calls. Allowed tools:
        - execute_python: implement data preparation, modeling, inference, and export submission.csv
        - validate_submission: validate the submission against sample_submission if available
        """
        system_instructions = (
            "You are a senior data analyst planning a Kaggle solution. "
            "Create a rigorous, feasible plan that can be executed via tool-calls. "
            "Return ONLY a JSON object with the following schema: {goal, assumptions, risks, notes, context, steps}.\n"
            "Each step must include: id, title, objective, rationale, tool_call {name, params}, dependencies, expected_outputs.\n"
            "Allowed tools: ['execute_python', 'validate_submission']. "
            "Prefer fewer steps but cover data loading, preprocessing, feature engineering, model training/validation, inference, and submission validation. "
            "Use file names already downloaded locally (train/test/sample_submission)."
        )

        user_context = self._build_user_context(problem_description, info, data_summary)

        prompt = f"{system_instructions}\n\n{user_context}\n\nReturn ONLY JSON."

        try:
            resp = self.llm.generate(prompt)
            text = resp.content
            plan_json = self._extract_json(text)
            plan_dict = json.loads(plan_json)
            plan_obj = self._to_plan(plan_dict)
            logger.info("Planning completed via LLM")
            return plan_obj
        except Exception as e:
            logger.warning(f"Failed to parse LLM planning output: {e}. Falling back to default plan.")
            return self._fallback_plan(info)

    def _build_user_context(self, problem_description: str, info: CompetitionInfo, data_summary: str) -> str:
        train_files = ", ".join(info.train_files) if info.train_files else "(none)"
        test_files = ", ".join(info.test_files) if info.test_files else "(none)"
        sample_submission = info.sample_submission_file or "(none)"

        return (
            "# Problem\n" + problem_description + "\n\n" +
            "# Competition\n" +
            f"id: {info.competition_id}\n" +
            f"type: {info.problem_type}\n" +
            f"metric: {info.evaluation_metric}\n" +
            f"train_files: {train_files}\n" +
            f"test_files: {test_files}\n" +
            f"sample_submission: {sample_submission}\n\n" +
            "# Data Summary\n" + data_summary
        )

    def _extract_json(self, text: str) -> str:
        """Extract a JSON object from LLM output, supporting fenced blocks."""
        m = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: first JSON-like object
        m2 = re.search(r"\{[\s\S]*\}", text)
        if m2:
            return m2.group(0)
        return text.strip()

    def _to_plan(self, plan_dict: Dict[str, Any]) -> Plan:
        steps: List[PlanStep] = []
        for step in plan_dict.get("steps", []):
            tc = step.get("tool_call") or {}
            tool = ToolCall(
                name=tc.get("name", "execute_python"),
                params=tc.get("params", {}),
                description=tc.get("description")
            ) if tc else None

            steps.append(
                PlanStep(
                    id=str(step.get("id", len(steps) + 1)),
                    title=step.get("title", "Step"),
                    objective=step.get("objective", ""),
                    rationale=step.get("rationale"),
                    tool_call=tool,
                    dependencies=list(step.get("dependencies", [])),
                    expected_outputs=list(step.get("expected_outputs", [])),
                )
            )

        return Plan(
            goal=plan_dict.get("goal", "Solve the Kaggle competition"),
            context=plan_dict.get("context", {}),
            steps=steps,
            assumptions=list(plan_dict.get("assumptions", [])),
            risks=list(plan_dict.get("risks", [])),
            notes=plan_dict.get("notes"),
        )

    def _fallback_plan(self, info: CompetitionInfo) -> Plan:
        """Provide a reasonable default plan when LLM parsing fails."""
        steps = [
            PlanStep(
                id="1",
                title="Prepare data",
                objective="Load training/test data and basic preprocessing (missing values, types).",
                rationale="Establish clean data for modeling.",
                tool_call=ToolCall(name="execute_python", params={}),
                expected_outputs=["Cleaned training and test frames"],
            ),
            PlanStep(
                id="2",
                title="Feature engineering",
                objective="Create meaningful features relevant to the competition.",
                rationale="Improve model signal.",
                tool_call=ToolCall(name="execute_python", params={}),
                expected_outputs=["Feature set ready for training"],
            ),
            PlanStep(
                id="3",
                title="Model training",
                objective="Train a strong baseline model with cross-validation.",
                rationale="Build reliable predictor.",
                tool_call=ToolCall(name="execute_python", params={}),
                expected_outputs=["Fitted model and CV metrics"],
            ),
            PlanStep(
                id="4",
                title="Inference and submission",
                objective="Run inference on test set and save 'submission.csv' in the working directory.",
                rationale="Produce competition submission.",
                tool_call=ToolCall(name="execute_python", params={}),
                expected_outputs=["submission.csv"],
            ),
            PlanStep(
                id="5",
                title="Validate submission",
                objective="Validate the generated 'submission.csv' against the sample submission if available.",
                rationale="Ensure format correctness.",
                tool_call=ToolCall(name="validate_submission", params={}),
                expected_outputs=["Validation report"],
            ),
        ]

        return Plan(
            goal=f"Solve {info.competition_id} with a solid baseline",
            context={"competition_id": info.competition_id},
            steps=steps,
            assumptions=["Data already downloaded locally", "Sample submission may exist"],
            risks=["Data leakage", "Overfitting", "Mismatched submission format"],
            notes="Fallback plan generated without LLM parsing.",
        )