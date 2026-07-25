"""
实验运行器
管理和执行RAG实验
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import random

from ..utils.logger import LoggerMixin, setup_logger
from ..utils.config import get_config
from ..utils.data_types import TraceableAnswer, ExperimentConfig

from .metrics import MetricsCalculator
from .llm_judge import LLMJudge


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_name: str
    method: str
    query_id: str
    query: str
    answer: str
    metrics: Dict[str, float]
    llm_judgments: Dict[str, Any]
    ground_truth: Optional[str]
    generation_time: float
    timestamp: str


class ExperimentRunner(LoggerMixin):
    """
    实验运行器
    管理完整的实验流程
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        use_llm_judge: bool = True
    ):
        config = get_config()
        self.output_dir = Path(output_dir or config.paths.get("results_dir", "./experiments/results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = MetricsCalculator()
        self.llm_judge = LLMJudge() if use_llm_judge else None

        self.logger = setup_logger(
            "experiment_runner",
            log_dir=str(config.paths.get("logs_dir", "./experiments/logs"))
        )

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        method_fn: Callable[[str], TraceableAnswer],
        queries: List[Dict[str, Any]],
        method_name: str
    ) -> List[ExperimentResult]:
        """
        运行单个方法的实验

        Args:
            experiment_config: 实验配置
            method_fn: 方法函数，接受query返回TraceableAnswer
            queries: 查询列表 [{"query": str, "ground_truth": str}, ...]
            method_name: 方法名称

        Returns:
            实验结果列表
        """
        self.logger.info(f"Running experiment: {experiment_config.experiment_name} - {method_name}")

        results = []
        total_queries = len(queries)

        for idx, query_data in enumerate(queries):
            query = query_data["query"]
            ground_truth = query_data.get("ground_truth")
            query_id = query_data.get("id", f"q_{idx}")

            self.logger.info(f"Processing query {idx + 1}/{total_queries}: {query[:50]}...")

            try:
                # 执行方法
                start_time = time.time()
                answer = method_fn(query)
                execution_time = time.time() - start_time

                # 计算指标
                metrics = self.metrics_calculator.calculate_all(
                    answer,
                    ground_truth=ground_truth
                )

                # LLM评判
                llm_judgments = {}
                if self.llm_judge:
                    llm_judgments = self.llm_judge.judge_all(answer, ground_truth)

                result = ExperimentResult(
                    experiment_name=experiment_config.experiment_name,
                    method=method_name,
                    query_id=query_id,
                    query=query,
                    answer=answer.answer,
                    metrics=metrics,
                    llm_judgments=llm_judgments,
                    ground_truth=ground_truth,
                    generation_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                continue

        self.logger.info(f"Completed {len(results)} queries for {method_name}")

        return results

    def run_comparison(
        self,
        experiment_config: ExperimentConfig,
        methods: Dict[str, Callable[[str], TraceableAnswer]],
        queries: List[Dict[str, Any]]
    ) -> Dict[str, List[ExperimentResult]]:
        """
        运行多方法对比实验

        Args:
            experiment_config: 实验配置
            methods: 方法字典 {method_name: method_fn}
            queries: 查询列表

        Returns:
            {method_name: [ExperimentResult, ...]}
        """
        all_results = {}

        for method_name, method_fn in methods.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running method: {method_name}")
            self.logger.info(f"{'='*50}")

            results = self.run_experiment(
                experiment_config,
                method_fn,
                queries,
                method_name
            )
            all_results[method_name] = results

        return all_results

    def save_results(
        self,
        results: Dict[str, List[ExperimentResult]],
        experiment_name: str
    ) -> str:
        """
        保存实验结果

        Args:
            results: 实验结果
            experiment_name: 实验名称

        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # 转换为可序列化格式
        output = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "results": {
                method: [asdict(r) for r in method_results]
                for method, method_results in results.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to {filepath}")

        return str(filepath)

    def aggregate_results(
        self,
        results: Dict[str, List[ExperimentResult]]
    ) -> Dict[str, Dict[str, float]]:
        """
        聚合实验结果

        Args:
            results: 实验结果

        Returns:
            聚合后的指标 {method: {metric: avg_value}}
        """
        aggregated = {}

        for method, method_results in results.items():
            if not method_results:
                continue

            # 收集所有指标
            all_metrics = {}
            for result in method_results:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

                # 添加LLM评判分数
                for judge_name, judgment in result.llm_judgments.items():
                    if isinstance(judgment, dict) and "score" in judgment:
                        metric_name = f"llm_{judge_name}"
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(judgment["score"])

            # 计算平均值
            aggregated[method] = {
                metric: sum(values) / len(values)
                for metric, values in all_metrics.items()
                if values
            }

            # 添加生成时间统计
            gen_times = [r.generation_time for r in method_results]
            aggregated[method]["avg_generation_time"] = sum(gen_times) / len(gen_times)

        return aggregated

    def print_comparison_table(
        self,
        aggregated: Dict[str, Dict[str, float]],
        metrics_to_show: Optional[List[str]] = None
    ) -> str:
        """
        打印对比表格

        Args:
            aggregated: 聚合结果
            metrics_to_show: 要显示的指标

        Returns:
            表格字符串
        """
        if not aggregated:
            return "No results to display"

        # 确定要显示的指标
        all_metrics = set()
        for method_metrics in aggregated.values():
            all_metrics.update(method_metrics.keys())

        if metrics_to_show:
            metrics = [m for m in metrics_to_show if m in all_metrics]
        else:
            metrics = sorted(all_metrics)

        # 构建表格
        lines = []
        header = "| Method | " + " | ".join(metrics) + " |"
        separator = "|" + "-" * (len("Method") + 2) + "|" + "|".join(["-" * 12 for _ in metrics]) + "|"

        lines.append(header)
        lines.append(separator)

        for method, method_metrics in sorted(aggregated.items()):
            values = []
            for metric in metrics:
                value = method_metrics.get(metric, 0)
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            row = f"| {method} | " + " | ".join(values) + " |"
            lines.append(row)

        table = "\n".join(lines)
        print(table)

        return table
