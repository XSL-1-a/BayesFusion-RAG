"""
图像检索器
支持基于CLIP的图像检索和基于摘要的文本检索
"""

from typing import List, Optional, Union
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..utils.embedder import Embedder, CLIPEmbedder
from ..utils.data_types import ImageElement, RetrievalResult, ModalityType, SourceMetadata, RetrievalLevel
from .indexer import VectorIndex, IndexItem


class ImageRetriever(LoggerMixin):
    """
    图像检索器
    支持两种检索模式：
    1. CLIP多模态检索：使用图像嵌入
    2. 摘要文本检索：使用图像摘要的文本嵌入
    """

    def __init__(
        self,
        index: Optional[VectorIndex] = None,
        mode: str = "clip"  # clip 或 summary
    ):
        self.index = index
        self.mode = mode
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            if self.mode == "clip":
                self._embedder = Embedder.get_image_embedder()
            else:
                self._embedder = Embedder.get_text_embedder()
        return self._embedder

    def set_index(self, index: VectorIndex) -> None:
        """设置索引"""
        self.index = index

    def retrieve(
        self,
        query: Union[str, Path],
        k: int = 5,
        filter_doc_ids: Optional[List[str]] = None,
        filter_section_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        检索相关图像

        Args:
            query: 查询文本或图像路径
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
        if isinstance(query, str) and not Path(query).exists():
            # 文本查询
            if self.mode == "clip":
                # CLIP文本嵌入
                query_embedding = self.embedder.embed_text(query)
            else:
                # 使用文本嵌入器
                text_embedder = Embedder.get_text_embedder()
                query_embedding = text_embedder.embed_text(query)
        elif isinstance(query, (str, Path)) and Path(query).exists():
            # 图像查询（图像搜图像）
            query_embedding = self.embedder.embed_image(query)
        else:
            # 默认使用文本
            if hasattr(self.embedder, 'embed_text'):
                query_embedding = self.embedder.embed_text(str(query))
            else:
                text_embedder = Embedder.get_text_embedder()
                query_embedding = text_embedder.embed_text(str(query))

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
            k=k * 2,
            filter_fn=filter_fn if (filter_doc_ids or filter_section_ids) else None
        )

        # 转换为RetrievalResult
        retrieval_results = []
        for item, score in results:
            if score < min_score:
                continue

            result = RetrievalResult(
                item_id=item.item_id,
                content=item.content,  # 图像摘要
                modality=ModalityType.IMAGE,
                score=score,
                source=SourceMetadata(
                    document=item.metadata.get("doc_id", ""),
                    page=item.metadata.get("page", 0),
                    section=item.metadata.get("section_id", "")
                ),
                level=RetrievalLevel.ELEMENT,
                retrieval_path=[
                    item.metadata.get("doc_id", ""),
                    item.metadata.get("section_id", ""),
                    item.item_id
                ],
                metadata={
                    **item.metadata,
                    "image_path": item.metadata.get("image_path", ""),
                    "image_type": item.metadata.get("image_type", "unknown")
                }
            )
            retrieval_results.append(result)

            if len(retrieval_results) >= k:
                break

        return retrieval_results

    def retrieve_by_similarity_to_image(
        self,
        image_path: Union[str, Path],
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        检索与给定图像相似的图像

        Args:
            image_path: 参考图像路径
            k: 返回数量

        Returns:
            RetrievalResult列表
        """
        if self.mode != "clip":
            self.logger.warning("Image-to-image retrieval requires CLIP mode")
            return []

        return self.retrieve(image_path, k=k)
