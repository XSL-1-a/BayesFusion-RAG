"""
评估指标计算模块
计算答案质量、检索质量、忠实度等指标
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ..utils.logger import LoggerMixin
from ..utils.data_types import TraceableAnswer, RetrievalResult


class MetricsCalculator(LoggerMixin):
    """
    评估指标计算器
    支持多种RAG评估指标
    """

    def calculate_all(
        self,
        answer: TraceableAnswer,
        ground_truth: Optional[str] = None,
        ground_truth_sources: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            answer: TraceableAnswer对象
            ground_truth: 标准答案（可选）
            ground_truth_sources: 标准来源（可选）

        Returns:
            指标字典
        """
        metrics = {}

        # 引用指标
        citation_metrics = self.calculate_citation_metrics(answer)
        metrics.update(citation_metrics)

        # 检索指标
        retrieval_metrics = self.calculate_retrieval_metrics(answer.sources)
        metrics.update(retrieval_metrics)

        # 如果有标准答案，计算答案质量
        if ground_truth:
            answer_metrics = self.calculate_answer_metrics(
                answer.answer,
                ground_truth
            )
            metrics.update(answer_metrics)

        return metrics

    def calculate_citation_metrics(
        self,
        answer: TraceableAnswer
    ) -> Dict[str, float]:
        """
        计算引用相关指标

        Args:
            answer: TraceableAnswer对象

        Returns:
            引用指标字典
        """
        citations = answer.citations

        if not citations:
            return {
                "citation_count": 0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "citation_f1": 0.0,
                "evidence_accuracy": 0.0
            }

        total = len(citations)
        verified = sum(1 for c in citations if c.verified)
        has_evidence = sum(1 for c in citations if c.evidence)

        precision = verified / total if total > 0 else 0

        # 简化的recall（基于引用覆盖）
        sources_cited = len(set(c.citation_id for c in citations))
        total_sources = len(answer.sources)
        recall = sources_cited / total_sources if total_sources > 0 else 0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "citation_count": total,
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": f1,
            "evidence_accuracy": has_evidence / total if total > 0 else 0
        }

    def calculate_retrieval_metrics(
        self,
        sources: List[RetrievalResult],
        k: int = 4
    ) -> Dict[str, float]:
        """
        计算检索相关指标

        Args:
            sources: 检索结果列表
            k: 计算指标的k值

        Returns:
            检索指标字典
        """
        if not sources:
            return {
                "avg_relevance": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0
            }

        scores = [s.score for s in sources[:k]]

        # 平均相关性
        avg_relevance = np.mean(scores)

        # MRR（简化版，假设第一个相关结果的倒数）
        # 这里用分数作为相关性判断
        mrr = 0.0
        for i, score in enumerate(scores):
            if score > 0.5:  # 阈值
                mrr = 1.0 / (i + 1)
                break

        # NDCG（简化版）
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        ndcg = dcg / idcg if idcg > 0 else 0

        return {
            "avg_relevance": avg_relevance,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def calculate_answer_metrics(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        计算答案质量指标（需要参考答案）

        Args:
            generated: 生成的答案
            reference: 参考答案

        Returns:
            答案质量指标
        """
        # 词级别的简单指标
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())

        # 移除停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}

        gen_words = gen_words - stop_words
        ref_words = ref_words - stop_words

        if not ref_words:
            return {"word_overlap": 0.0, "answer_coverage": 0.0}

        overlap = gen_words & ref_words
        precision = len(overlap) / len(gen_words) if gen_words else 0
        recall = len(overlap) / len(ref_words) if ref_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "word_overlap": f1,
            "answer_precision": precision,
            "answer_recall": recall
        }

    def calculate_hallucination_rate(
        self,
        answer: TraceableAnswer
    ) -> float:
        """
        估计幻觉率
        基于未验证引用的比例

        Args:
            answer: TraceableAnswer对象

        Returns:
            幻觉率估计
        """
        if not answer.citations:
            # 无引用时，假设高幻觉风险
            return 0.5

        unverified = sum(1 for c in answer.citations if not c.verified)
        rate = unverified / len(answer.citations)

        return rate
