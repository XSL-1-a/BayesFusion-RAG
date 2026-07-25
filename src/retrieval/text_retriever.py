"""
文本检索器
基于向量相似度的文本检索
"""

from typing import List, Optional, Dict, Any, Tuple

from ..utils.logger import LoggerMixin
from ..utils.embedder import Embedder
from ..utils.data_types import Chunk, RetrievalResult, ModalityType, SourceMetadata, RetrievalLevel
from .indexer import VectorIndex, IndexItem


class TextRetriever(LoggerMixin):
    """
    文本检索器
    支持基于向量相似度的文本块检索
    """

    def __init__(
        self,
        index: Optional[VectorIndex] = None,
        embedder=None
    ):
        self.index = index
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder.get_text_embedder()
        return self._embedder

    def set_index(self, index: VectorIndex) -> None:
        """设置索引"""
        self.index = index

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_doc_ids: Optional[List[str]] = None,
        filter_section_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        检索相关文本块

        Args:
            query: 查询文本
            k: 返回数量
            filter_doc_ids: 限制在这些文档内检索
            filter_section_ids: 限制在这些章节内检索
            min_score: 最小相似度阈值

        Returns:
            RetrievalResult列表
        """
        if self.index is None:
            self.logger.error("Index not set")
            return []

        # 嵌入查询
        query_embedding = self.embedder.embed_text(query)

        # 构建过滤函数
        def filter_fn(item: IndexItem) -> bool:
            if filter_doc_ids and item.metadata.get("doc_id") not in filter_doc_ids:
                return False
            if filter_section_ids and item.metadata.get("section_id") not in filter_section_ids:
                return False
            return True

        # 搜索
        results = self.index.search(
            query_embedding,
            k=k * 2,  # 过检索
            filter_fn=filter_fn if (filter_doc_ids or filter_section_ids) else None
        )

        # 转换为RetrievalResult
        retrieval_results = []
        for item, score in results:
            if score < min_score:
                continue

            result = RetrievalResult(
                item_id=item.item_id,
                content=item.content,
                modality=ModalityType.TEXT,
                score=score,
                source=SourceMetadata(
                    document=item.metadata.get("doc_id", ""),
                    page=item.metadata.get("page", 0),
                    section=item.metadata.get("section_title", "")
                ),
                level=RetrievalLevel.ELEMENT,
                retrieval_path=[
                    item.metadata.get("doc_id", ""),
                    item.metadata.get("section_id", ""),
                    item.item_id
                ],
                metadata=item.metadata
            )
            retrieval_results.append(result)

            if len(retrieval_results) >= k:
                break

        return retrieval_results

    def retrieve_by_ids(
        self,
        item_ids: List[str]
    ) -> List[RetrievalResult]:
        """
        根据ID检索文本块

        Args:
            item_ids: 文本块ID列表

        Returns:
            RetrievalResult列表
        """
        if self.index is None:
            return []

        results = []
        for item_id in item_ids:
            if item_id in self.index._id_to_idx:
                idx = self.index._id_to_idx[item_id]
                item = self.index.items[idx]
                result = RetrievalResult(
                    item_id=item.item_id,
                    content=item.content,
                    modality=ModalityType.TEXT,
                    score=1.0,
                    source=SourceMetadata(
                        document=item.metadata.get("doc_id", ""),
                        page=item.metadata.get("page", 0),
                        section=item.metadata.get("section_title", "")
                    ),
                    level=RetrievalLevel.ELEMENT,
                    metadata=item.metadata
                )
                results.append(result)

        return results
