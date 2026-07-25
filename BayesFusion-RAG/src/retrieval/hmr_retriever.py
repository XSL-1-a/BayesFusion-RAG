"""
层次化多模态检索器 (HMR - Hierarchical Multimodal Retrieval)
创新点1：三级级联检索
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.config import get_config, HMRConfig
from ..utils.embedder import Embedder
from ..utils.data_types import RetrievalResult, ModalityType, SourceMetadata, RetrievalLevel

from .indexer import HierarchicalIndexer, VectorIndex, IndexItem
from .text_retriever import TextRetriever
from .image_retriever import ImageRetriever
from .multimodal_retriever import MultimodalRetriever


@dataclass
class HMRResult:
    """HMR检索结果"""
    results: List[RetrievalResult]
    retrieval_path: Dict[str, Any]  # 检索路径信息
    stats: Dict[str, Any]  # 统计信息


class HierarchicalMultimodalRetriever(LoggerMixin):
    """
    层次化多模态检索器

    三级检索流程：
    Level 1: 文档级粗筛 -> Top-N 文档
    Level 2: 章节级定位 -> Top-M 章节
    Level 3: 元素级精检 -> Top-K 元素（文本+图像）
    """

    def __init__(
        self,
        config: Optional[HMRConfig] = None,
        indexer: Optional[HierarchicalIndexer] = None
    ):
        self.config = config or get_config().hmr
        self.indexer = indexer or HierarchicalIndexer(self.config)

        # 嵌入器
        self._text_embedder = None

        # 各级检索器
        self.doc_retriever: Optional[TextRetriever] = None
        self.section_retriever: Optional[TextRetriever] = None
        self.element_retriever: Optional[MultimodalRetriever] = None

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            self._text_embedder = Embedder.get_text_embedder()
        return self._text_embedder

    def initialize(self) -> None:
        """初始化检索器"""
        # 加载索引
        self.indexer.load_index()

        # 初始化各级检索器
        if self.indexer.doc_index:
            self.doc_retriever = TextRetriever(self.indexer.doc_index)

        if self.indexer.section_index:
            self.section_retriever = TextRetriever(self.indexer.section_index)

        if self.indexer.text_index:
            self.element_retriever = MultimodalRetriever(
                text_index=self.indexer.text_index,
                image_index=self.indexer.image_index
            )

        self.logger.info("HMR Retriever initialized")

    def retrieve(
        self,
        query: str,
        top_n_docs: Optional[int] = None,
        top_m_sections: Optional[int] = None,
        top_k_elements: Optional[int] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5
    ) -> HMRResult:
        """
        层次化多模态检索

        Args:
            query: 查询文本
            top_n_docs: 文档级检索数量
            top_m_sections: 章节级检索数量
            top_k_elements: 元素级检索数量
            text_weight: 文本权重
            image_weight: 图像权重

        Returns:
            HMRResult对象
        """
        # 使用配置默认值
        top_n = top_n_docs or self.config.top_n_docs
        top_m = top_m_sections or self.config.top_m_sections
        top_k = top_k_elements or self.config.top_k_elements

        retrieval_path = {}
        stats = {"levels": {}}

        # Level 1: 文档级粗筛
        self.logger.debug(f"Level 1: Retrieving top-{top_n} documents")
        doc_results = self._retrieve_documents(query, top_n)
        doc_ids = [r.item_id for r in doc_results]

        retrieval_path["level_1_docs"] = doc_ids
        stats["levels"]["document"] = {
            "retrieved": len(doc_ids),
            "top_scores": [r.score for r in doc_results[:3]]
        }

        # Level 2: 章节级定位
        self.logger.debug(f"Level 2: Retrieving top-{top_m} sections from {len(doc_ids)} docs")
        section_results = self._retrieve_sections(query, top_m, doc_ids)
        section_ids = [r.item_id for r in section_results]

        retrieval_path["level_2_sections"] = section_ids
        stats["levels"]["section"] = {
            "retrieved": len(section_ids),
            "top_scores": [r.score for r in section_results[:3]]
        }

        # Level 3: 元素级精检
        self.logger.debug(f"Level 3: Retrieving top-{top_k} elements from {len(section_ids)} sections")
        element_results = self._retrieve_elements(
            query,
            top_k,
            section_ids,
            text_weight,
            image_weight
        )

        retrieval_path["level_3_elements"] = [r.item_id for r in element_results]
        stats["levels"]["element"] = {
            "retrieved": len(element_results),
            "text_count": sum(1 for r in element_results if r.modality == ModalityType.TEXT),
            "image_count": sum(1 for r in element_results if r.modality == ModalityType.IMAGE)
        }

        # 为结果添加完整的检索路径
        for result in element_results:
            result.retrieval_path = [
                result.source.document,
                result.metadata.get("section_id", ""),
                result.item_id
            ]

        return HMRResult(
            results=element_results,
            retrieval_path=retrieval_path,
            stats=stats
        )

    def _retrieve_documents(
        self,
        query: str,
        k: int
    ) -> List[RetrievalResult]:
        """Level 1: 文档级检索"""
        if self.doc_retriever is None:
            self.logger.warning("Document retriever not initialized")
            return []

        results = self.doc_retriever.retrieve(query, k=k)

        # 转换level
        for r in results:
            r.level = RetrievalLevel.DOCUMENT

        return results

    def _retrieve_sections(
        self,
        query: str,
        k: int,
        doc_ids: List[str]
    ) -> List[RetrievalResult]:
        """Level 2: 章节级检索"""
        if self.section_retriever is None:
            self.logger.warning("Section retriever not initialized")
            return []

        results = self.section_retriever.retrieve(
            query,
            k=k,
            filter_doc_ids=doc_ids
        )

        # 转换level
        for r in results:
            r.level = RetrievalLevel.SECTION

        return results

    def _retrieve_elements(
        self,
        query: str,
        k: int,
        section_ids: List[str],
        text_weight: float,
        image_weight: float
    ) -> List[RetrievalResult]:
        """Level 3: 元素级检索"""
        if self.element_retriever is None:
            self.logger.warning("Element retriever not initialized")
            return []

        results = self.element_retriever.retrieve(
            query,
            k=k,
            text_weight=text_weight,
            image_weight=image_weight,
            filter_section_ids=section_ids
        )

        return results

    def retrieve_flat(
        self,
        query: str,
        k: int = 4,
        text_weight: float = 0.5,
        image_weight: float = 0.5
    ) -> List[RetrievalResult]:
        """
        扁平检索（用于基线对比）
        直接在元素级检索，不使用层次结构

        Args:
            query: 查询文本
            k: 返回数量
            text_weight: 文本权重
            image_weight: 图像权重

        Returns:
            RetrievalResult列表
        """
        if self.element_retriever is None:
            self.logger.warning("Element retriever not initialized")
            return []

        return self.element_retriever.retrieve(
            query,
            k=k,
            text_weight=text_weight,
            image_weight=image_weight
        )

    def get_retrieval_statistics(self, hmr_result: HMRResult) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "total_retrieved": len(hmr_result.results),
            "retrieval_path": hmr_result.retrieval_path,
            "level_stats": hmr_result.stats["levels"],
            "modality_distribution": {
                "text": sum(1 for r in hmr_result.results if r.modality == ModalityType.TEXT),
                "image": sum(1 for r in hmr_result.results if r.modality == ModalityType.IMAGE)
            },
            "score_range": {
                "min": min(r.score for r in hmr_result.results) if hmr_result.results else 0,
                "max": max(r.score for r in hmr_result.results) if hmr_result.results else 0,
                "mean": sum(r.score for r in hmr_result.results) / len(hmr_result.results) if hmr_result.results else 0
            }
        }
