from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class ToolCall:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class PlanStep:
    id: str
    title: str
    objective: str
    rationale: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    dependencies: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    steps: List[PlanStep] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class ExecutionArtifact:
    step_id: str
    success: bool
    stdout: str = ""
    stderr: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    code: Optional[str] = None 


@dataclass
class ToolResult:
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    success: bool
    plan: Plan
    artifacts: List[ExecutionArtifact] = field(default_factory=list)
    submission_path: Optional[Path] = None
    validation: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    llm_calls: int = 0
    total_time_seconds: float = 0.0
    generated_code: Optional[str] = None