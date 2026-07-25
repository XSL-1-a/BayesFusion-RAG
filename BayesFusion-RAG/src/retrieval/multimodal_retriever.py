"""
多模态检索器
整合文本和图像检索
"""

from typing import List, Optional, Tuple

from ..utils.logger import LoggerMixin
from ..utils.data_types import RetrievalResult, ModalityType
from .text_retriever import TextRetriever
from .image_retriever import ImageRetriever
from .indexer import VectorIndex


class MultimodalRetriever(LoggerMixin):
    """
    多模态检索器
    同时检索文本和图像，支持结果融合
    """

    def __init__(
        self,
        text_index: Optional[VectorIndex] = None,
        image_index: Optional[VectorIndex] = None,
        text_k: int = 2,
        image_k: int = 2,
        image_mode: str = "summary"  # clip 或 summary
    ):
        self.text_retriever = TextRetriever(text_index)
        self.image_retriever = ImageRetriever(image_index, mode=image_mode)
        self.text_k = text_k
        self.image_k = image_k

    def set_indices(
        self,
        text_index: VectorIndex,
        image_index: Optional[VectorIndex] = None
    ) -> None:
        """设置索引"""
        self.text_retriever.set_index(text_index)
        if image_index:
            self.image_retriever.set_index(image_index)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        filter_doc_ids: Optional[List[str]] = None,
        filter_section_ids: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        多模态检索

        Args:
            query: 查询文本
            k: 总返回数量
            text_weight: 文本权重
            image_weight: 图像权重
            filter_doc_ids: 文档过滤
            filter_section_ids: 章节过滤

        Returns:
            RetrievalResult列表
        """
        # 计算各模态的k值
        total_weight = text_weight + image_weight
        k_text = max(1, int(k * text_weight / total_weight))
        k_image = k - k_text

        # 文本检索
        text_results = self.text_retriever.retrieve(
            query,
            k=k_text * 2,  # 过检索
            filter_doc_ids=filter_doc_ids,
            filter_section_ids=filter_section_ids
        )

        # 图像检索
        image_results = []
        if k_image > 0 and self.image_retriever.index is not None:
            image_results = self.image_retriever.retrieve(
                query,
                k=k_image * 2,
                filter_doc_ids=filter_doc_ids,
                filter_section_ids=filter_section_ids
            )

        # 合并结果
        combined = self._merge_results(
            text_results,
            image_results,
            k,
            text_weight,
            image_weight
        )

        return combined

    def _merge_results(
        self,
        text_results: List[RetrievalResult],
        image_results: List[RetrievalResult],
        k: int,
        text_weight: float,
        image_weight: float
    ) -> List[RetrievalResult]:
        """
        融合检索结果

        Args:
            text_results: 文本检索结果
            image_results: 图像检索结果
            k: 返回数量
            text_weight: 文本权重
            image_weight: 图像权重

        Returns:
            融合后的结果
        """
        # 调整分数
        for result in text_results:
            result.score *= text_weight

        for result in image_results:
            result.score *= image_weight

        # 合并并排序
        all_results = text_results + image_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # 去重（基于item_id）
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.item_id not in seen_ids:
                seen_ids.add(result.item_id)
                unique_results.append(result)
                if len(unique_results) >= k:
                    break

        return unique_results

    def retrieve_with_allocation(
        self,
        query: str,
        k_text: int,
        k_image: int,
        filter_doc_ids: Optional[List[str]] = None,
        filter_section_ids: Optional[List[str]] = None
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        按指定数量分别检索文本和图像

        Args:
            query: 查询文本
            k_text: 文本检索数量
            k_image: 图像检索数量
            filter_doc_ids: 文档过滤
            filter_section_ids: 章节过滤

        Returns:
            (文本结果, 图像结果)
        """
        text_results = self.text_retriever.retrieve(
            query,
            k=k_text,
            filter_doc_ids=filter_doc_ids,
            filter_section_ids=filter_section_ids
        )

        image_results = []
        if k_image > 0 and self.image_retriever.index is not None:
            image_results = self.image_retriever.retrieve(
                query,
                k=k_image,
                filter_doc_ids=filter_doc_ids,
                filter_section_ids=filter_section_ids
            )

        return text_results, image_results
