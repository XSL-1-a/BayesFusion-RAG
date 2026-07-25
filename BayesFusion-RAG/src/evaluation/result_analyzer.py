"""
结果分析器
分析和可视化实验结果
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from ..utils.logger import LoggerMixin


class ResultAnalyzer(LoggerMixin):
    """
    结果分析器
    分析实验结果，生成统计报告
    """

    def __init__(self, results_dir: str = "./experiments/results"):
        self.results_dir = Path(results_dir)

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """加载实验结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_methods(
        self,
        results: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比不同方法的性能

        Args:
            results: 实验结果
            metrics: 要对比的指标

        Returns:
            对比结果
        """
        method_results = results.get("results", {})
        comparison = {}

        for method, method_data in method_results.items():
            if not method_data:
                continue

            # 收集指标
            metric_values = {}
            for result in method_data:
                for metric, value in result.get("metrics", {}).items():
                    if metrics and metric not in metrics:
                        continue
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)

            # 计算统计
            comparison[method] = {}
            for metric, values in metric_values.items():
                comparison[method][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }

        return comparison

    def generate_report(
        self,
        comparison: Dict[str, Dict[str, Any]],
        experiment_name: str
    ) -> str:
        """
        生成实验报告

        Args:
            comparison: 对比结果
            experiment_name: 实验名称

        Returns:
            报告Markdown
        """
        lines = [
            f"# Experiment Report: {experiment_name}",
            "",
            "## Summary",
            ""
        ]

        # 方法概览
        lines.append("### Methods Evaluated")
        for method in comparison.keys():
            lines.append(f"- {method}")
        lines.append("")

        # 指标对比
        lines.append("## Metrics Comparison")
        lines.append("")

        # 获取所有指标
        all_metrics = set()
        for method_stats in comparison.values():
            all_metrics.update(method_stats.keys())

        for metric in sorted(all_metrics):
            lines.append(f"### {metric}")
            lines.append("")
            lines.append("| Method | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")

            for method, stats in comparison.items():
                if metric in stats:
                    s = stats[metric]
                    lines.append(
                        f"| {method} | {s['mean']:.4f} | {s['std']:.4f} | "
                        f"{s['min']:.4f} | {s['max']:.4f} |"
                    )

            lines.append("")

        # 最佳方法
        lines.append("## Best Performing Method by Metric")
        lines.append("")

        for metric in sorted(all_metrics):
            best_method = None
            best_score = -float('inf')

            for method, stats in comparison.items():
                if metric in stats:
                    if stats[metric]['mean'] > best_score:
                        best_score = stats[metric]['mean']
                        best_method = method

            if best_method:
                lines.append(f"- **{metric}**: {best_method} ({best_score:.4f})")

        return "\n".join(lines)

    def compute_improvement(
        self,
        comparison: Dict[str, Dict[str, Any]],
        baseline_method: str,
        target_method: str
    ) -> Dict[str, float]:
        """
        计算相对提升

        Args:
            comparison: 对比结果
            baseline_method: 基线方法
            target_method: 目标方法

        Returns:
            各指标的提升百分比
        """
        if baseline_method not in comparison or target_method not in comparison:
            return {}

        baseline = comparison[baseline_method]
        target = comparison[target_method]

        improvements = {}
        for metric in baseline:
            if metric in target:
                base_val = baseline[metric]["mean"]
                target_val = target[metric]["mean"]

                if base_val != 0:
                    improvement = (target_val - base_val) / abs(base_val) * 100
                    improvements[metric] = improvement

        return improvements

    def statistical_significance_test(
        self,
        results: Dict[str, Any],
        method1: str,
        method2: str,
        metric: str
    ) -> Dict[str, Any]:
        """
        统计显著性检验

        Args:
            results: 实验结果
            method1: 方法1
            method2: 方法2
            metric: 指标名称

        Returns:
            检验结果
        """
        try:
            from scipy import stats
        except ImportError:
            return {"error": "scipy not installed"}

        method_results = results.get("results", {})

        values1 = [
            r["metrics"].get(metric, 0)
            for r in method_results.get(method1, [])
        ]
        values2 = [
            r["metrics"].get(metric, 0)
            for r in method_results.get(method2, [])
        ]

        if not values1 or not values2:
            return {"error": "Insufficient data"}

        # t检验
        t_stat, p_value = stats.ttest_ind(values1, values2)

        return {
            "method1": method1,
            "method2": method2,
            "metric": metric,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "method1_mean": np.mean(values1),
            "method2_mean": np.mean(values2)
        }
