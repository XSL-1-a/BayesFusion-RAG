"""
检索反思模块
评估检索结果的相关性，过滤无关内容
"""

from typing import List, Optional, Dict, Any

from ..utils.logger import LoggerMixin
from ..utils.llm_client import LLMClient
from ..utils.data_types import RetrievalResult


class RetrievalReflection(LoggerMixin):
    """
    检索反思模块 (Inspired by mR²AG)
    对检索结果进行二次评估，过滤无关内容
    """

    REFLECTION_PROMPT = """Given the query and retrieved document, determine if this document is truly relevant to answering the query.

Query: {query}

Document content:
{document}

Document metadata:
- Source: {source}
- Page: {page}
- Modality: {modality}

Evaluate:
1. Does this document contain information directly relevant to the query?
2. Would this document help answer the query?
3. Is the information specific enough to be useful?

Output a JSON object:
{{
    "relevant": true or false,
    "relevance_score": 0.0-1.0,
    "reason": "brief explanation",
    "key_information": "what relevant info is in this document (if any)"
}}"""

    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def filter_relevant(
        self,
        query: str,
        documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        过滤相关文档

        Args:
            query: 用户查询
            documents: 检索结果列表

        Returns:
            过滤后的相关文档
        """
        relevant_docs = []

        for doc in documents:
            reflection = self.reflect(query, doc)

            if reflection.get("relevant", False):
                # 更新分数
                doc.score = doc.score * reflection.get("relevance_score", 0.5)
                doc.metadata["reflection"] = {
                    "reason": reflection.get("reason", ""),
                    "key_info": reflection.get("key_information", "")
                }
                relevant_docs.append(doc)

        self.logger.debug(
            f"Reflection: {len(relevant_docs)}/{len(documents)} documents kept"
        )

        return relevant_docs

    def reflect(
        self,
        query: str,
        document: RetrievalResult
    ) -> Dict[str, Any]:
        """
        对单个文档进行反思评估

        Args:
            query: 用户查询
            document: 检索结果

        Returns:
            反思结果字典
        """
        prompt = self.REFLECTION_PROMPT.format(
            query=query,
            document=document.content[:2000],
            source=document.source.document,
            page=document.source.page,
            modality=document.modality.value
        )

        try:
            response = self.llm_client.generate_structured(prompt)
            return response
        except Exception as e:
            self.logger.warning(f"Reflection failed: {e}")
            # 默认保留
            return {
                "relevant": True,
                "relevance_score": document.score,
                "reason": "Reflection failed, keeping document"
            }

    def batch_reflect(
        self,
        query: str,
        documents: List[RetrievalResult],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        批量反思（未来可以并行化）

        Args:
            query: 用户查询
            documents: 检索结果列表
            max_concurrent: 最大并发数

        Returns:
            反思结果列表
        """
        results = []
        for doc in documents:
            result = self.reflect(query, doc)
            results.append(result)
        return results

    def compute_relevance_feedback(
        self,
        reflections: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        计算相关性反馈统计

        Args:
            reflections: 反思结果列表

        Returns:
            相关性统计
        """
        if not reflections:
            return {"text": 0.5, "image": 0.5}

        relevant_count = sum(1 for r in reflections if r.get("relevant", False))
        avg_score = sum(r.get("relevance_score", 0) for r in reflections) / len(reflections)

        return {
            "relevant_ratio": relevant_count / len(reflections),
            "avg_relevance": avg_score,
            "total_evaluated": len(reflections)
        }
