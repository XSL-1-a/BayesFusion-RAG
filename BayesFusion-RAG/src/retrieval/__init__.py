"""
层次化多模态检索模块 (HMR - Hierarchical Multimodal Retrieval)
- 三级层次索引构建 (文档级 -> 章节级 -> 元素级)
- 多模态嵌入 (文本 + 图像)
- 级联检索策略
"""

from .indexer import HierarchicalIndexer
from .text_retriever import TextRetriever
from .image_retriever import ImageRetriever
from .multimodal_retriever import MultimodalRetriever
from .hmr_retriever import HierarchicalMultimodalRetriever

__all__ = [
    "HierarchicalIndexer",
    "TextRetriever",
    "ImageRetriever",
    "MultimodalRetriever",
    "HierarchicalMultimodalRetriever"
]
