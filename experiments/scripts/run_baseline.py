#!/usr/bin/env python3
"""
基线实验脚本
运行原论文的基线方法
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
from typing import List, Dict, Any

from src.utils.config import get_config
from src.utils.data_types import TraceableAnswer, ExperimentConfig
from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.hmr_retriever import HierarchicalMultimodalRetriever
from src.retrieval.indexer import HierarchicalIndexer
from src.generation.generator import TraceableAnswerGenerator
from src.evaluation.experiment_runner import ExperimentRunner


def run_baseline_experiment(
    data_dir: str,
    queries: List[Dict[str, Any]],
    output_dir: str = "./experiments/results"
):
    """
    运行基线实验

    Args:
        data_dir: 数据目录
        queries: 查询列表
        output_dir: 输出目录
    """
    config = get_config()

    # 初始化组件
    print("Initializing components...")

    # 文档处理
    processor = DocumentProcessor(output_dir=f"{data_dir}/processed")

    # 索引器
    indexer = HierarchicalIndexer(index_dir=f"{data_dir}/embeddings")

    # 检索器（扁平检索 - 基线）
    retriever = HierarchicalMultimodalRetriever(indexer=indexer)
    retriever.initialize()

    # 生成器
    generator = TraceableAnswerGenerator()

    # 定义基线方法
    def baseline_flat_retrieval(query: str) -> TraceableAnswer:
        """扁平检索基线"""
        results = retriever.retrieve_flat(query, k=4)
        return generator.generate(query, results)

    def baseline_text_only(query: str) -> TraceableAnswer:
        """仅文本检索基线"""
        results = retriever.retrieve_flat(query, k=4, text_weight=1.0, image_weight=0.0)
        return generator.generate(query, results)

    def baseline_multimodal_fixed(query: str) -> TraceableAnswer:
        """多模态固定k=2+2基线"""
        text_results = retriever.element_retriever.text_retriever.retrieve(query, k=2)
        image_results = retriever.element_retriever.image_retriever.retrieve(query, k=2)
        results = text_results + image_results
        return generator.generate(query, results)

    # 实验配置
    experiment_config = ExperimentConfig(
        experiment_name="baseline_comparison",
        description="Baseline methods comparison",
        methods=["flat", "text_only", "multimodal_fixed"],
        dataset="industrial_docs",
        metrics=["answer_correctness", "context_relevancy", "citation_precision"]
    )

    # 运行实验
    runner = ExperimentRunner(output_dir=output_dir)

    methods = {
        "flat_retrieval": baseline_flat_retrieval,
        "text_only": baseline_text_only,
        "multimodal_fixed_k2": baseline_multimodal_fixed
    }

    results = runner.run_comparison(experiment_config, methods, queries)

    # 保存结果
    result_file = runner.save_results(results, "baseline")

    # 输出汇总
    aggregated = runner.aggregate_results(results)
    print("\n" + "=" * 50)
    print("BASELINE EXPERIMENT RESULTS")
    print("=" * 50)
    runner.print_comparison_table(aggregated)

    return result_file


if __name__ == "__main__":
    # 示例查询
    sample_queries = [
        {
            "id": "q1",
            "query": "What is the operating temperature range of the device?",
            "ground_truth": "The operating temperature range is -40°C to +85°C."
        },
        {
            "id": "q2",
            "query": "How to perform the calibration procedure?",
            "ground_truth": "The calibration procedure involves..."
        }
    ]

    run_baseline_experiment(
        data_dir="./data",
        queries=sample_queries
    )
