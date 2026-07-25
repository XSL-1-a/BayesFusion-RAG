#!/usr/bin/env python3
"""
完整实验脚本
运行所有方法（基线 + 创新点）的对比实验
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from typing import List, Dict, Any

from src.utils.config import get_config
from src.utils.data_types import TraceableAnswer, ExperimentConfig
from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.hmr_retriever import HierarchicalMultimodalRetriever
from src.retrieval.indexer import HierarchicalIndexer
from src.entity.entity_enhanced_rag import EntityEnhancedRAG
from src.fusion.amf_retriever import AdaptiveModalityFusion
from src.generation.generator import TraceableAnswerGenerator
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.result_analyzer import ResultAnalyzer


class HEAMRAGSystem:
    """
    HEA-MRAG完整系统
    整合所有创新点
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        # 初始化索引器
        self.indexer = HierarchicalIndexer(
            index_dir=str(self.data_dir / "embeddings")
        )

        # HMR检索器
        self.hmr_retriever = HierarchicalMultimodalRetriever(indexer=self.indexer)

        # AMF融合检索器
        self.amf_retriever = AdaptiveModalityFusion()

        # 实体增强
        self.entity_rag = EntityEnhancedRAG()

        # 生成器
        self.generator = TraceableAnswerGenerator()

        self._initialized = False

    def initialize(self):
        """初始化系统"""
        if self._initialized:
            return

        print("Loading indices...")
        self.indexer.load_index()
        self.hmr_retriever.initialize()

        # 设置AMF的索引
        if self.indexer.text_index:
            self.amf_retriever.set_indices(
                self.indexer.text_index,
                self.indexer.image_index
            )

        self._initialized = True
        print("System initialized")

    # ========== 基线方法 ==========

    def baseline_no_rag(self, query: str) -> TraceableAnswer:
        """基线1：无RAG（直接LLM）"""
        from src.utils.llm_client import LLMClient
        llm = LLMClient.get_client()
        answer = llm.generate(f"Answer this question: {query}")
        return TraceableAnswer(
            query=query,
            answer=answer,
            citations=[],
            sources=[],
            confidence=0.3,
            generation_time=0,
            metadata={"method": "no_rag"}
        )

    def baseline_flat_retrieval(self, query: str) -> TraceableAnswer:
        """基线2：扁平检索"""
        results = self.hmr_retriever.retrieve_flat(query, k=4)
        return self.generator.generate(query, results)

    def baseline_fixed_k(self, query: str) -> TraceableAnswer:
        """基线3：固定k=2+2多模态检索"""
        results = self.amf_retriever.retrieve_with_fixed_k(query, k_text=2, k_image=2)
        return self.generator.generate(query, results)

    # ========== 创新方法 ==========

    def method_hmr_only(self, query: str) -> TraceableAnswer:
        """创新点1：仅HMR"""
        hmr_result = self.hmr_retriever.retrieve(query)
        return self.generator.generate(query, hmr_result.results)

    def method_hmr_ea(self, query: str) -> TraceableAnswer:
        """创新点1+2：HMR + 实体增强"""
        hmr_result = self.hmr_retriever.retrieve(query)

        # 实体增强
        query_entities = self.entity_rag.extract_query_entities(query)
        enhanced_results = self.entity_rag.enhance_context_with_entities(
            hmr_result.results,
            query_entities
        )

        return self.generator.generate(query, enhanced_results)

    def method_hmr_ea_amf(self, query: str) -> TraceableAnswer:
        """创新点1+2+3：HMR + EA + AMF"""
        # 使用AMF进行自适应检索
        amf_result = self.amf_retriever.retrieve(query, total_k=6)

        # 然后通过HMR过滤
        doc_ids = list(set(r.source.document for r in amf_result.results))
        hmr_result = self.hmr_retriever.retrieve(
            query,
            top_n_docs=min(3, len(doc_ids)),
            top_k_elements=4
        )

        # 实体增强
        query_entities = self.entity_rag.extract_query_entities(query)
        enhanced_results = self.entity_rag.enhance_context_with_entities(
            hmr_result.results,
            query_entities
        )

        return self.generator.generate(query, enhanced_results)

    def method_full(self, query: str) -> TraceableAnswer:
        """完整方法：HMR + EA + AMF + TAG"""
        # AMF自适应检索
        amf_result = self.amf_retriever.retrieve(query, total_k=6)

        # HMR层次化过滤
        doc_ids = list(set(r.source.document for r in amf_result.results))
        hmr_result = self.hmr_retriever.retrieve(
            query,
            top_n_docs=min(3, len(doc_ids)),
            top_k_elements=4
        )

        # 实体增强
        query_entities = self.entity_rag.extract_query_entities(query)
        enhanced_results = self.entity_rag.enhance_context_with_entities(
            hmr_result.results,
            query_entities
        )

        # TAG可追溯生成
        answer = self.generator.generate_with_rejection(
            query,
            enhanced_results,
            min_confidence=0.4
        )

        return answer


def run_full_experiment(
    data_dir: str,
    queries: List[Dict[str, Any]],
    output_dir: str = "./experiments/results"
):
    """
    运行完整对比实验

    Args:
        data_dir: 数据目录
        queries: 查询列表
        output_dir: 输出目录
    """
    # 初始化系统
    system = HEAMRAGSystem(data_dir)
    system.initialize()

    # 实验配置
    experiment_config = ExperimentConfig(
        experiment_name="hea_mrag_full_comparison",
        description="HEA-MRAG: Full ablation study comparing baseline and innovations",
        methods=[
            "no_rag",
            "flat_retrieval",
            "fixed_k",
            "hmr_only",
            "hmr_ea",
            "hmr_ea_amf",
            "full"
        ],
        dataset="industrial_docs",
        metrics=[
            "answer_correctness",
            "context_relevancy",
            "citation_precision",
            "citation_recall",
            "faithfulness"
        ]
    )

    # 定义方法映射
    methods = {
        "B1_no_rag": system.baseline_no_rag,
        "B2_flat_retrieval": system.baseline_flat_retrieval,
        "B3_fixed_k2+2": system.baseline_fixed_k,
        "Ours_HMR": system.method_hmr_only,
        "Ours_HMR+EA": system.method_hmr_ea,
        "Ours_HMR+EA+AMF": system.method_hmr_ea_amf,
        "Ours_Full": system.method_full
    }

    # 运行实验
    runner = ExperimentRunner(output_dir=output_dir, use_llm_judge=True)
    results = runner.run_comparison(experiment_config, methods, queries)

    # 保存结果
    result_file = runner.save_results(results, "hea_mrag_full")

    # 分析结果
    analyzer = ResultAnalyzer(output_dir)
    comparison = analyzer.compare_methods({"results": results})

    # 生成报告
    report = analyzer.generate_report(comparison, "HEA-MRAG Full Experiment")

    report_file = Path(output_dir) / "experiment_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # 输出汇总
    print("\n" + "=" * 60)
    print("HEA-MRAG FULL EXPERIMENT RESULTS")
    print("=" * 60)

    aggregated = runner.aggregate_results(results)
    runner.print_comparison_table(aggregated)

    # 计算相对提升
    print("\n" + "-" * 60)
    print("IMPROVEMENTS OVER BASELINE")
    print("-" * 60)

    improvements = analyzer.compute_improvement(
        comparison,
        "B2_flat_retrieval",
        "Ours_Full"
    )

    for metric, improvement in improvements.items():
        sign = "+" if improvement > 0 else ""
        print(f"  {metric}: {sign}{improvement:.1f}%")

    print(f"\nResults saved to: {result_file}")
    print(f"Report saved to: {report_file}")

    return result_file


if __name__ == "__main__":
    # 加载查询数据
    queries_file = Path("./data/queries.json")

    if queries_file.exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
    else:
        # 示例查询
        queries = [
            {
                "id": "q1",
                "query": "What is the operating temperature range?",
                "ground_truth": "The operating temperature range is -40°C to +85°C."
            },
            {
                "id": "q2",
                "query": "Describe the calibration procedure.",
                "ground_truth": "The calibration procedure consists of..."
            },
            {
                "id": "q3",
                "query": "What safety constraints apply to the installation?",
                "ground_truth": "Safety constraints include..."
            }
        ]

    run_full_experiment(
        data_dir="./data",
        queries=queries
    )
