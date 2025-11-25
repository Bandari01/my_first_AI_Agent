"""
Evaluation Metrics Calculator

Calculates various performance metrics for AI agents
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from backend.agents.base_agent import AgentResult, AgentType
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMetrics:
    """
    AI Agent Evaluation Metrics
    
    Contains evaluation metrics across multiple dimensions
    """
    # Basic Info
    agent_type: str
    competition_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Performance Metrics
    total_time: float = 0.0  # Total time (seconds)
    code_generation_time: float = 0.0  # Code generation time
    execution_time: float = 0.0  # Execution time
    
    # Efficiency Metrics
    llm_calls: int = 0  # Number of LLM calls
    llm_tokens: int = 0  # Number of tokens used (if available)
    code_lines: int = 0  # Number of generated code lines
    iterations: int = 0  # Number of iterations
    
    # Quality Metrics
    success: bool = False  # Whether submission was generated successfully
    code_quality_score: float = 0.0  # Code quality score (0-100)
    submission_valid: bool = False  # Whether submission is valid
    
    # Complexity Metrics
    code_complexity: float = 0.0  # Code complexity
    feature_count: int = 0  # Number of features used
    model_complexity: str = "unknown"  # Model complexity (simple/medium/complex)
    
    # Autonomy Metrics
    autonomy_score: float = 0.0  # Autonomy score (0-100)
    human_interventions: int = 0  # Number of human interventions required
    
    # Explainability Metrics
    explainability_score: float = 0.0  # Explainability score (0-100)
    comments_ratio: float = 0.0  # Ratio of comments
    thoughts_count: int = 0  # Number of recorded thought steps
    
    # Resource Consumption
    memory_peak_mb: float = 0.0  # Peak memory usage (MB)
    cpu_usage_percent: float = 0.0  # CPU usage percentage
    
    # Errors and Warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Extra Info
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_type": self.agent_type,
            "competition_name": self.competition_name,
            "timestamp": self.timestamp,
            "total_time": self.total_time,
            "code_generation_time": self.code_generation_time,
            "execution_time": self.execution_time,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "code_lines": self.code_lines,
            "iterations": self.iterations,
            "success": self.success,
            "code_quality_score": self.code_quality_score,
            "submission_valid": self.submission_valid,
            "code_complexity": self.code_complexity,
            "feature_count": self.feature_count,
            "model_complexity": self.model_complexity,
            "autonomy_score": self.autonomy_score,
            "human_interventions": self.human_interventions,
            "explainability_score": self.explainability_score,
            "comments_ratio": self.comments_ratio,
            "thoughts_count": self.thoughts_count,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "errors": self.errors,
            "warnings": self.warnings,
            "extra_metrics": self.extra_metrics
        }
    
    def save(self, path: Path):
        """Save metrics to file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_overall_score(self) -> float:
        """
        Calculate overall score (0-100)
        
        Weight distribution:
        - Success: 30%
        - Efficiency: 25%
        - Quality: 25%
        - Autonomy: 10%
        - Explainability: 10%
        """
        scores = []
        
        # 1. Success (30%)
        success_score = 100.0 if self.success and self.submission_valid else 0.0
        scores.append(success_score * 0.30)
        
        # 2. Efficiency (25%)
        # The shorter the time, the better (assuming 300 seconds as baseline)
        time_score = max(0, 100 - (self.total_time / 300) * 100)
        # The fewer LLM calls, the better (assuming 10 calls as baseline)
        llm_score = max(0, 100 - (self.llm_calls / 10) * 100)
        efficiency_score = (time_score + llm_score) / 2
        scores.append(efficiency_score * 0.25)
        
        # 3. Quality (25%)
        scores.append(self.code_quality_score * 0.25)
        
        # 4. Autonomy (10%)
        scores.append(self.autonomy_score * 0.10)
        
        # 5. Explainability (10%)
        scores.append(self.explainability_score * 0.10)
        
        return sum(scores)


class MetricsCalculator:
    """
    Metrics Calculator
    
    Calculates various evaluation metrics from AgentResult
    """
    
    def __init__(self):
        """Initialize Calculator"""
        logger.info("Initializing MetricsCalculator")
    
    def calculate(self, result: AgentResult) -> AgentMetrics:
        """
        Calculate complete metrics from AgentResult
        
        Args:
            result: Agent execution result
            
        Returns:
            Calculated metrics
        """
        logger.info(f"Starting to calculate metrics: {result.agent_type.value}")
        
        metrics = AgentMetrics(
            agent_type=result.agent_type.value,
            competition_name=result.competition_name
        )
        
        # Performance Metrics
        metrics.total_time = result.total_time
        metrics.code_generation_time = result.code_generation_time
        metrics.execution_time = result.execution_time
        
        # Efficiency Metrics
        metrics.llm_calls = result.llm_calls
        metrics.code_lines = result.code_lines
        metrics.iterations = len(result.actions)
        
        # Quality Metrics
        metrics.success = (result.submission_file_path is not None)
        metrics.submission_valid = (result.execution_error is None)
        
        # Analyze Code Quality
        if result.generated_code:
            metrics.code_quality_score = self._calculate_code_quality(result.generated_code)
            metrics.code_complexity = self._calculate_complexity(result.generated_code)
            metrics.comments_ratio = self._calculate_comments_ratio(result.generated_code)
            metrics.model_complexity = self._infer_model_complexity(result.generated_code)
        
        # Autonomy Score
        metrics.autonomy_score = self._calculate_autonomy(result)
        
        # Explainability Score
        metrics.explainability_score = self._calculate_explainability(result)
        metrics.thoughts_count = len(result.thoughts)
        
        # Errors and Warnings
        if result.error_message:
            metrics.errors.append(result.error_message)
        if result.execution_error:
            metrics.errors.append(result.execution_error)
        
        logger.info(f"✓ Metrics calculation completed, overall score: {metrics.get_overall_score():.2f}")
        
        return metrics
    
    def _calculate_code_quality(self, code: str) -> float:
        """
        Calculate code quality score (0-100)
        
        Factors considered:
        - Reasonable code length
        - Has comments
        - Has error handling
        - Has log output
        - Modularity
        """
        score = 50.0  # Base score
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 1. Reasonable length (+10 points)
        if 50 <= len(non_empty_lines) <= 500:
            score += 10
        
        # 2. Has comments (+15 points)
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if len(comment_lines) >= 5:
            score += 15
        elif len(comment_lines) >= 2:
            score += 8
        
        # 3. Has error handling (+10 points)
        if 'try:' in code or 'except' in code:
            score += 10
        
        # 4. Has log/print output (+5 points)
        if 'print(' in code or 'logger' in code:
            score += 5
        
        # 5. Has function definition (modularity) (+10 points)
        if 'def ' in code:
            score += 10
        
        # 6. Imported common libraries (+10 points)
        common_imports = ['pandas', 'numpy', 'sklearn']
        import_count = sum(1 for lib in common_imports if lib in code)
        score += min(10, import_count * 3)
        
        return min(100.0, score)
    
    def _calculate_complexity(self, code: str) -> float:
        """
        Calculate code complexity (simplified version of cyclomatic complexity)
        
        Returns:
            Complexity score (lower is better)
        """
        complexity = 1.0  # Base complexity
        
        # Control flow statements increase complexity
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except']
        for keyword in control_keywords:
            complexity += code.count(f' {keyword} ') + code.count(f'\n{keyword} ')
        
        return complexity
    
    def _calculate_comments_ratio(self, code: str) -> float:
        """Calculate comments ratio"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        if len(non_empty_lines) == 0:
            return 0.0
        
        return len(comment_lines) / len(non_empty_lines)
    
    def _infer_model_complexity(self, code: str) -> str:
        """
        Infer model complexity
        
        Returns:
            "simple", "medium", "complex"
        """
        # Simple models
        simple_models = ['LinearRegression', 'LogisticRegression', 'DecisionTree']
        # Medium models
        medium_models = ['RandomForest', 'GradientBoosting', 'SVM', 'KNN']
        # Complex models
        complex_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Neural', 'LSTM', 'Transformer']
        
        for model in complex_models:
            if model in code:
                return "complex"
        
        for model in medium_models:
            if model in code:
                return "medium"
        
        for model in simple_models:
            if model in code:
                return "simple"
        
        return "unknown"
    
    def _calculate_autonomy(self, result: AgentResult) -> float:
        """
        Calculate autonomy score (0-100)
        
        Factors considered:
        - Whether human intervention is required
        - Whether all steps are completed automatically
        - Error recovery capability
        """
        score = 100.0
        
        # If there is an error, reduce autonomy score
        if result.error_message:
            score -= 30
        
        # If no submission is generated, autonomy is insufficient
        if not result.submission_file_path:
            score -= 40
        
        # If too many iterations, it means multiple attempts were needed
        if result.llm_calls > 10:
            score -= 10
        
        return max(0.0, score)
    
    def _calculate_explainability(self, result: AgentResult) -> float:
        """
        Calculate explainability score (0-100)
        
        Factors considered:
        - Whether there are thought process records
        - Code comment quality
        - Whether there are intermediate outputs
        """
        score = 0.0
        
        # 1. Has thought records (+40 points)
        if len(result.thoughts) > 0:
            score += min(40, len(result.thoughts) * 10)
        
        # 2. Has action records (+20 points)
        if len(result.actions) > 0:
            score += min(20, len(result.actions) * 5)
        
        # 3. Has observation records (+20 points)
        if len(result.observations) > 0:
            score += min(20, len(result.observations) * 5)
        
        # 4. Code has comments (+20 points)
        if result.generated_code:
            comment_ratio = self._calculate_comments_ratio(result.generated_code)
            score += comment_ratio * 100 * 0.2
        
        return min(100.0, score)
    
    def calculate_batch(self, results: List[AgentResult]) -> List[AgentMetrics]:
        """
        Batch calculate metrics
        
        Args:
            results: List of AgentResult
            
        Returns:
            List of metrics
        """
        logger.info(f"Batch calculating metrics: {len(results)} results")
        metrics_list = []
        
        for i, result in enumerate(results, 1):
            logger.info(f"Processing {i}/{len(results)}: {result.agent_type.value}")
            metrics = self.calculate(result)
            metrics_list.append(metrics)
        
        logger.info("✓ Batch calculation completed")
        return metrics_list

