"""
AI Agent Comparator

Compares performance of different agent architectures
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from backend.evaluation.metrics import AgentMetrics
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonReport:
    """
    Comparison Report
    
    Contains performance comparison results of multiple agents
    """
    competition_name: str
    agents: List[str] = field(default_factory=list)  # Agents participating in comparison
    metrics_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Rankings
    rankings: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Rankings for each metric
    overall_ranking: List[tuple[str, float]] = field(default_factory=list)  # Overall ranking
    
    # Best Performance
    best_performer: Dict[str, str] = field(default_factory=dict)  # Best agent for each metric
    
    # Statistical Analysis
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Conclusions and Recommendations
    conclusions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "competition_name": self.competition_name,
            "agents": self.agents,
            "metrics_comparison": self.metrics_comparison,
            "rankings": self.rankings,
            "overall_ranking": self.overall_ranking,
            "best_performer": self.best_performer,
            "statistics": self.statistics,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations
        }
    
    def save(self, path: Path):
        """Save report"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Generate Markdown report"""
        lines = []
        lines.append(f"# AI Agent Performance Comparison Report")
        lines.append(f"\n## Competition: {self.competition_name}")
        lines.append(f"\nAgents compared: {', '.join(self.agents)}")
        
        # Overall Ranking
        lines.append(f"\n## Overall Ranking")
        lines.append("\n| Rank | Agent Type | Overall Score |")
        lines.append("|------|------------|---------------|")
        for i, (agent, score) in enumerate(self.overall_ranking, 1):
            lines.append(f"| {i} | {agent} | {score:.2f} |")
        
        # Detailed Metrics Comparison
        lines.append(f"\n## Detailed Metrics Comparison")
        
        if self.metrics_comparison:
            # Select key metrics to display
            key_metrics = [
                "total_time", "code_generation_time", "execution_time",
                "llm_calls", "code_lines", "code_quality_score",
                "autonomy_score", "explainability_score"
            ]
            
            for metric in key_metrics:
                if metric in self.metrics_comparison:
                    lines.append(f"\n### {metric}")
                    lines.append("\n| Agent | Value |")
                    lines.append("|-------|-------|")
                    for agent, value in self.metrics_comparison[metric].items():
                        lines.append(f"| {agent} | {value:.2f} |")
        
        # Best Performance
        lines.append(f"\n## Best Agent by Metric")
        lines.append("\n| Metric | Best Agent |")
        lines.append("|--------|------------|")
        for metric, agent in self.best_performer.items():
            lines.append(f"| {metric} | {agent} |")
        
        # Conclusions
        if self.conclusions:
            lines.append(f"\n## Conclusions")
            for conclusion in self.conclusions:
                lines.append(f"\n- {conclusion}")
        
        # Recommendations
        if self.recommendations:
            lines.append(f"\n## Recommendations")
            for recommendation in self.recommendations:
                lines.append(f"\n- {recommendation}")
        
        return "\n".join(lines)


class AgentComparator:
    """
    Agent Comparator
    
    Compares performance metrics of multiple agents
    """
    
    def __init__(self):
        """Initialize Comparator"""
        logger.info("Initializing AgentComparator")
    
    def compare(
        self,
        metrics_list: List[AgentMetrics],
        competition_name: Optional[str] = None
    ) -> ComparisonReport:
        """
        Compare metrics of multiple agents
        
        Args:
            metrics_list: List of metrics
            competition_name: Competition name
            
        Returns:
            Comparison report
        """
        if len(metrics_list) < 2:
            logger.warning("At least 2 agents are needed for comparison")
        
        logger.info(f"Starting to compare {len(metrics_list)} agents")
        
        # Create report
        report = ComparisonReport(
            competition_name=competition_name or metrics_list[0].competition_name,
            agents=[m.agent_type for m in metrics_list]
        )
        
        # 1. Extract and compare metrics
        report.metrics_comparison = self._extract_metrics(metrics_list)
        
        # 2. Calculate rankings
        report.rankings = self._calculate_rankings(report.metrics_comparison)
        
        # 3. Calculate overall ranking
        report.overall_ranking = self._calculate_overall_ranking(metrics_list)
        
        # 4. Find best performers
        report.best_performer = self._find_best_performers(report.metrics_comparison)
        
        # 5. Generate statistical analysis
        report.statistics = self._generate_statistics(report.metrics_comparison)
        
        # 6. Generate conclusions
        report.conclusions = self._generate_conclusions(report)
        
        # 7. Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        logger.info("✓ Comparison completed")
        return report
    
    def _extract_metrics(
        self,
        metrics_list: List[AgentMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Extract all metric values"""
        comparison = {}
        
        # Get all metric names
        metric_names = [
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_lines", "code_quality_score",
            "code_complexity", "autonomy_score", "explainability_score",
            "comments_ratio", "thoughts_count"
        ]
        
        for metric_name in metric_names:
            comparison[metric_name] = {}
            for metrics in metrics_list:
                value = getattr(metrics, metric_name, 0)
                comparison[metric_name][metrics.agent_type] = float(value)
        
        return comparison
    
    def _calculate_rankings(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate rankings for each metric
        
        Returns:
            {metric_name: {agent_type: rank}}
        """
        rankings = {}
        
        # Define which metrics are better when lower
        lower_is_better = {
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_complexity"
        }
        
        for metric_name, values in metrics_comparison.items():
            # Sort
            sorted_agents = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=(metric_name not in lower_is_better)
            )
            
            # Assign rank
            rankings[metric_name] = {}
            for rank, (agent, _) in enumerate(sorted_agents, 1):
                rankings[metric_name][agent] = rank
        
        return rankings
    
    def _calculate_overall_ranking(
        self,
        metrics_list: List[AgentMetrics]
    ) -> List[tuple[str, float]]:
        """
        Calculate overall ranking
        
        Returns:
            [(agent_type, overall_score)], sorted by score descending
        """
        scores = []
        for metrics in metrics_list:
            overall_score = metrics.get_overall_score()
            scores.append((metrics.agent_type, overall_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _find_best_performers(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """Find best agent for each metric"""
        best_performers = {}
        
        lower_is_better = {
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_complexity"
        }
        
        for metric_name, values in metrics_comparison.items():
            if not values:
                continue
            
            if metric_name in lower_is_better:
                best_agent = min(values.items(), key=lambda x: x[1])[0]
            else:
                best_agent = max(values.items(), key=lambda x: x[1])[0]
            
            best_performers[metric_name] = best_agent
        
        return best_performers
    
    def _generate_statistics(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Generate statistics"""
        statistics = {}
        
        for metric_name, values in metrics_comparison.items():
            if not values:
                continue
            
            values_list = list(values.values())
            statistics[metric_name] = {
                "min": min(values_list),
                "max": max(values_list),
                "mean": sum(values_list) / len(values_list),
                "range": max(values_list) - min(values_list)
            }
        
        return statistics
    
    def _generate_conclusions(self, report: ComparisonReport) -> List[str]:
        """Generate conclusions"""
        conclusions = []
        
        if report.overall_ranking:
            best_agent = report.overall_ranking[0][0]
            best_score = report.overall_ranking[0][1]
            conclusions.append(
                f"{best_agent} performed best, overall score {best_score:.2f}"
            )
        
        # Analyze time efficiency
        if "total_time" in report.metrics_comparison:
            times = report.metrics_comparison["total_time"]
            fastest = min(times.items(), key=lambda x: x[1])
            slowest = max(times.items(), key=lambda x: x[1])
            conclusions.append(
                f"{fastest[0]} fastest ({fastest[1]:.2f}s), "
                f"{slowest[0]} slowest ({slowest[1]:.2f}s)"
            )
        
        # Analyze code quality
        if "code_quality_score" in report.metrics_comparison:
            quality = report.metrics_comparison["code_quality_score"]
            best_quality = max(quality.items(), key=lambda x: x[1])
            conclusions.append(
                f"{best_quality[0]} highest code quality ({best_quality[1]:.2f} points)"
            )
        
        # Analyze autonomy
        if "autonomy_score" in report.metrics_comparison:
            autonomy = report.metrics_comparison["autonomy_score"]
            most_autonomous = max(autonomy.items(), key=lambda x: x[1])
            conclusions.append(
                f"{most_autonomous[0]} most autonomous ({most_autonomous[1]:.2f} points)"
            )
        
        return conclusions
    
    def _generate_recommendations(self, report: ComparisonReport) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        # Recommendation based on overall ranking
        if report.overall_ranking:
            best_agent = report.overall_ranking[0][0]
            recommendations.append(
                f"For {report.competition_name} type problems, recommend using {best_agent} architecture"
            )
        
        # Based on time efficiency
        if "total_time" in report.best_performer:
            fastest = report.best_performer["total_time"]
            recommendations.append(
                f"If speed is priority, choose {fastest}"
            )
        
        # Based on code quality
        if "code_quality_score" in report.best_performer:
            best_quality = report.best_performer["code_quality_score"]
            recommendations.append(
                f"If code quality is priority, choose {best_quality}"
            )
        
        # Based on explainability
        if "explainability_score" in report.best_performer:
            most_explainable = report.best_performer["explainability_score"]
            recommendations.append(
                f"If high explainability is needed, choose {most_explainable}"
            )
        
        return recommendations
    
    def generate_visualization_data(
        self,
        report: ComparisonReport
    ) -> Dict[str, Any]:
        """
        Generate visualization data (for frontend plotting)
        
        Returns:
            Data structure suitable for Plotly or other visualization libraries
        """
        viz_data = {
            "radar_chart": self._prepare_radar_chart_data(report),
            "bar_chart": self._prepare_bar_chart_data(report),
            "time_comparison": self._prepare_time_chart_data(report),
        }
        
        return viz_data
    
    def _prepare_radar_chart_data(self, report: ComparisonReport) -> Dict:
        """Prepare radar chart data (comprehensive capability comparison)"""
        # Select key dimensions
        dimensions = [
            "code_quality_score",
            "autonomy_score",
            "explainability_score"
        ]
        
        data = {"agents": report.agents, "dimensions": [], "values": {}}
        
        for dim in dimensions:
            if dim in report.metrics_comparison:
                data["dimensions"].append(dim)
                for agent in report.agents:
                    if agent not in data["values"]:
                        data["values"][agent] = []
                    value = report.metrics_comparison[dim].get(agent, 0)
                    data["values"][agent].append(value)
        
        return data
    
    def _prepare_bar_chart_data(self, report: ComparisonReport) -> Dict:
        """Prepare bar chart data (overall score comparison)"""
        data = {
            "agents": [agent for agent, _ in report.overall_ranking],
            "scores": [score for _, score in report.overall_ranking]
        }
        return data
    
    def _prepare_time_chart_data(self, report: ComparisonReport) -> Dict:
        """Prepare time comparison chart data"""
        time_metrics = ["code_generation_time", "execution_time"]
        
        data = {"agents": report.agents, "metrics": [], "values": {}}
        
        for metric in time_metrics:
            if metric in report.metrics_comparison:
                data["metrics"].append(metric)
                for agent in report.agents:
                    if agent not in data["values"]:
                        data["values"][agent] = {}
                    data["values"][agent][metric] = report.metrics_comparison[metric].get(agent, 0)
        
        return data

