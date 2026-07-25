"""
跨模态重排序模块
对混合模态检索结果进行重排序
"""

from typing import List, Optional
from abc import ABC, abstractmethod

from ..utils.logger import LoggerMixin
from ..utils.llm_client import LLMClient
from ..utils.data_types import RetrievalResult, ModalityType


class BaseReranker(ABC, LoggerMixin):
    """重排序器基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: int = 4
    ) -> List[RetrievalResult]:
        """重排序文档"""
        pass


class ScoreBasedReranker(BaseReranker):
    """基于原始分数的重排序（基线）"""

    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: int = 4
    ) -> List[RetrievalResult]:
        """按原始分数排序"""
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]


class BM25Reranker(BaseReranker):
    """BM25重排序"""

    def __init__(self):
        self._bm25 = None

    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: int = 4
    ) -> List[RetrievalResult]:
        """使用BM25重排序"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.logger.warning("rank_bm25 not installed, falling back to score-based")
            return ScoreBasedReranker().rerank(query, documents, top_k)

        # 构建语料库
        corpus = [doc.content.split() for doc in documents]
        bm25 = BM25Okapi(corpus)

        # 计算BM25分数
        query_tokens = query.split()
        bm25_scores = bm25.get_scores(query_tokens)

        # 结合原始分数
        for i, doc in enumerate(documents):
            combined_score = 0.5 * doc.score + 0.5 * (bm25_scores[i] / max(bm25_scores))
            doc.score = combined_score
            doc.metadata["bm25_score"] = bm25_scores[i]

        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]


class LLMReranker(BaseReranker):
    """LLM重排序"""

    RERANK_PROMPT = """You are evaluating the relevance of retrieved documents for a query.

Query: {query}

Documents to evaluate:
{documents}

For each document, assess:
1. Direct relevance to the query
2. Information quality and specificity
3. Modality appropriateness (text vs image)

Output a JSON object with rankings:
{{
    "rankings": [
        {{
            "doc_id": "document identifier",
            "relevance_score": 0.0-1.0,
            "reason": "brief explanation"
        }}
    ]
}}

Rank from most to least relevant."""

    def __init__(self):
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: int = 4
    ) -> List[RetrievalResult]:
        """使用LLM重排序"""
        if len(documents) <= top_k:
            return documents

        # 构建文档描述
        doc_descriptions = []
        for i, doc in enumerate(documents):
            desc = f"""[Doc {i}] ({doc.modality.value})
Source: {doc.source.document}, Page {doc.source.page}
Content: {doc.content[:500]}..."""
            doc_descriptions.append(desc)

        prompt = self.RERANK_PROMPT.format(
            query=query,
            documents="\n\n".join(doc_descriptions)
        )

        try:
            response = self.llm_client.generate_structured(prompt)
            rankings = response.get("rankings", [])

            # 更新分数
            doc_scores = {str(i): doc.score for i, doc in enumerate(documents)}
            for rank in rankings:
                doc_id = str(rank.get("doc_id", "")).replace("Doc ", "")
                if doc_id.isdigit():
                    idx = int(doc_id)
                    if 0 <= idx < len(documents):
                        new_score = rank.get("relevance_score", 0.5)
                        # 结合原始分数
                        documents[idx].score = 0.3 * documents[idx].score + 0.7 * new_score
                        documents[idx].metadata["llm_rerank_reason"] = rank.get("reason", "")

        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}")

        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]


class CrossModalReranker(BaseReranker):
    """
    跨模态重排序器
    考虑模态多样性的重排序
    """

    def __init__(
        self,
        base_reranker: Optional[BaseReranker] = None,
        diversity_weight: float = 0.2
    ):
        self.base_reranker = base_reranker or ScoreBasedReranker()
        self.diversity_weight = diversity_weight

    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: int = 4
    ) -> List[RetrievalResult]:
        """
        跨模态重排序
        在保持相关性的同时确保模态多样性
        """
        # 先用基础重排序器
        ranked_docs = self.base_reranker.rerank(query, documents, top_k * 2)

        # 应用多样性重排序 (MMR-like)
        final_docs = []
        remaining = list(ranked_docs)

        while len(final_docs) < top_k and remaining:
            if not final_docs:
                # 第一个文档直接选最高分
                best = remaining[0]
            else:
                # 后续文档考虑多样性
                best = None
                best_score = -float('inf')

                for doc in remaining:
                    # 相关性分数
                    relevance = doc.score

                    # 多样性分数（与已选文档的模态差异）
                    diversity = self._compute_diversity(doc, final_docs)

                    # 综合分数
                    combined = (1 - self.diversity_weight) * relevance + self.diversity_weight * diversity

                    if combined > best_score:
                        best_score = combined
                        best = doc

            if best:
                final_docs.append(best)
                remaining.remove(best)

        return final_docs

    def _compute_diversity(
        self,
        candidate: RetrievalResult,
        selected: List[RetrievalResult]
    ) -> float:
        """计算候选文档与已选文档的多样性"""
        if not selected:
            return 1.0

        # 模态多样性
        candidate_modality = candidate.modality
        same_modality_count = sum(
            1 for doc in selected
            if doc.modality == candidate_modality
        )

        modality_diversity = 1.0 - (same_modality_count / len(selected))

        # 来源多样性
        candidate_source = candidate.source.document
        same_source_count = sum(
            1 for doc in selected
            if doc.source.document == candidate_source
        )

        source_diversity = 1.0 - (same_source_count / len(selected))

        return 0.6 * modality_diversity + 0.4 * source_diversity
