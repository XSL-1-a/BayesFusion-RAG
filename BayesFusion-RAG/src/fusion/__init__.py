"""
自适应模态融合模块 (AMF - Adaptive Modality Fusion)
- 查询模态偏好分类
- 动态k值分配
- 检索反思 (Retrieval Reflection)
- 跨模态重排序

MoRE (Mixture-of-Retrieval-Experts)
- 学习式门控网络
- 多检索专家混合
- N-Expert 最优权重理论
"""

from .query_classifier import QueryModalityClassifier
from .k_allocator import AdaptiveKAllocator
from .reflection import RetrievalReflection
from .reranker import CrossModalReranker
from .amf_retriever import AdaptiveModalityFusion

from .more_features import MoREFeatureExtractor
from .gating_network import MoREGatingNet, MoRELoss, MoRETrainer
from .retrieval_experts import (
    BaseExpert, BM25Expert, DenseExpert, EntityExpert, ImageExpert,
    create_default_experts
)
from .more_retriever import MoRERetriever

__all__ = [
    # AMF
    "QueryModalityClassifier",
    "AdaptiveKAllocator",
    "RetrievalReflection",
    "CrossModalReranker",
    "AdaptiveModalityFusion",
    # MoRE
    "MoREFeatureExtractor",
    "MoREGatingNet",
    "MoRELoss",
    "MoRETrainer",
    "BaseExpert",
    "BM25Expert",
    "DenseExpert",
    "EntityExpert",
    "ImageExpert",
    "create_default_experts",
    "MoRERetriever",
]
