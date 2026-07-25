"""
自适应模态融合检索器 (AMF - Adaptive Modality Fusion)
创新点3：根据查询特性动态调整模态融合策略
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.config import get_config, AMFConfig, MoREConfig
from ..utils.data_types import RetrievalResult, ModalityType

from ..retrieval.text_retriever import TextRetriever
from ..retrieval.image_retriever import ImageRetriever
from ..retrieval.indexer import VectorIndex

from .query_classifier import QueryModalityClassifier, ClassificationResult
from .k_allocator import AdaptiveKAllocator
from .reflection import RetrievalReflection
from .reranker import CrossModalReranker, LLMReranker, BaseReranker


@dataclass
class AMFResult:
    """AMF检索结果"""
    results: List[RetrievalResult]
    classification: ClassificationResult
    k_allocation: Tuple[int, int]
    reflection_stats: Dict[str, Any]
    rerank_applied: bool


class AdaptiveModalityFusion(LoggerMixin):
    """
    自适应模态融合检索器

    Pipeline:
    1. 查询分类：判断模态偏好
    2. K值分配：动态分配文本/图像检索数量
    3. 并行检索：分别检索文本和图像
    4. 检索反思：过滤无关结果
    5. 跨模态重排序：综合排序
    """

    def __init__(
        self,
        config: Optional[AMFConfig] = None,
        more_config: Optional[MoREConfig] = None,
        text_index: Optional[VectorIndex] = None,
        image_index: Optional[VectorIndex] = None
    ):
        self.config = config or get_config().amf
        self.more_config = more_config

        # MoRE 模式
        self._more_retriever = None
        if self.more_config and self.more_config.enabled:
            self._init_more_retriever()

        # 组件
        self.classifier = QueryModalityClassifier(
            confidence_threshold=self.config.confidence_threshold
        )
        self.k_allocator = AdaptiveKAllocator()
        self.reflection = RetrievalReflection() if self.config.reflection_enabled else None

        # 重排序器
        if self.config.reranker_method == "llm":
            base_reranker = LLMReranker()
        else:
            from .reranker import BM25Reranker
            base_reranker = BM25Reranker()
        self.reranker = CrossModalReranker(base_reranker=base_reranker)

        # 检索器
        self.text_retriever = TextRetriever(text_index)
        self.image_retriever = ImageRetriever(image_index)

    def _init_more_retriever(self):
        """初始化 MoRE 检索器"""
        try:
            from .more_retriever import MoRERetriever
            from .gating_network import MoREGatingNet
            from .retrieval_experts import create_default_experts

            cfg = self.more_config
            experts = create_default_experts()

            gating_net = None
            if cfg.mode == "learned" and cfg.checkpoint_path:
                gating_net = MoREGatingNet(
                    feature_dim=cfg.query_dim + 18,
                    n_experts=cfg.n_experts,
                    hidden_dim=cfg.hidden_dim,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k_gate
                )

            self._more_retriever = MoRERetriever(
                experts=experts,
                gating_net=gating_net,
                mode=cfg.mode,
                n_experts=cfg.n_experts,
                query_dim=cfg.query_dim
            )

            if cfg.checkpoint_path:
                self._more_retriever.load_gating_network(cfg.checkpoint_path)

            self.logger.info(f"MoRE retriever initialized (mode={cfg.mode})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MoRE retriever: {e}")
            self._more_retriever = None

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
        total_k: int = 4,
        filter_doc_ids: Optional[List[str]] = None,
        filter_section_ids: Optional[List[str]] = None
    ) -> AMFResult:
        """
        自适应模态融合检索

        Args:
            query: 用户查询
            total_k: 总检索数量
            filter_doc_ids: 文档过滤
            filter_section_ids: 章节过滤

        Returns:
            AMFResult对象
        """
        # Step 1: 查询分类
        self.logger.debug("Step 1: Query classification")
        classification = self.classifier.classify(query)
        self.logger.info(
            f"Query classified as {classification.preference.value} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Step 2: K值分配
        self.logger.debug("Step 2: K allocation")
        k_text, k_image = self.k_allocator.allocate(classification, total_k)
        self.logger.info(f"K allocation: text={k_text}, image={k_image}")

        # Step 3: 并行检索（过检索）
        self.logger.debug("Step 3: Parallel retrieval")
        text_results = self.text_retriever.retrieve(
            query,
            k=k_text * 2,  # 过检索用于反思
            filter_doc_ids=filter_doc_ids,
            filter_section_ids=filter_section_ids
        )

        image_results = []
        if k_image > 0 and self.image_retriever.index is not None:
            image_results = self.image_retriever.retrieve(
                query,
                k=k_image * 2,
                filter_doc_ids=filter_doc_ids,
                filter_section_ids=filter_section_ids
            )

        self.logger.debug(
            f"Retrieved: {len(text_results)} text, {len(image_results)} image"
        )

        # Step 4: 检索反思
        reflection_stats = {"applied": False}
        if self.reflection and self.config.reflection_enabled:
            self.logger.debug("Step 4: Retrieval reflection")

            text_results = self.reflection.filter_relevant(query, text_results)
            if image_results:
                image_results = self.reflection.filter_relevant(query, image_results)

            reflection_stats = {
                "applied": True,
                "text_kept": len(text_results),
                "image_kept": len(image_results)
            }
            self.logger.info(
                f"After reflection: {len(text_results)} text, {len(image_results)} image"
            )

        # Step 5: 合并并重排序
        self.logger.debug("Step 5: Cross-modal reranking")
        all_results = text_results[:k_text * 2] + image_results[:k_image * 2]

        reranked_results = self.reranker.rerank(query, all_results, top_k=total_k)

        # 确保模态分配大致符合预期
        final_results = self._balance_modalities(
            reranked_results,
            k_text,
            k_image,
            classification
        )

        return AMFResult(
            results=final_results,
            classification=classification,
            k_allocation=(k_text, k_image),
            reflection_stats=reflection_stats,
            rerank_applied=True
        )

    def _balance_modalities(
        self,
        results: List[RetrievalResult],
        target_text: int,
        target_image: int,
        classification: ClassificationResult
    ) -> List[RetrievalResult]:
        """
        平衡模态分配
        确保最终结果大致符合预期的模态比例
        """
        # 如果置信度低，不强制平衡
        if classification.confidence < 0.6:
            return results

        text_results = [r for r in results if r.modality == ModalityType.TEXT]
        image_results = [r for r in results if r.modality == ModalityType.IMAGE]

        total_k = target_text + target_image

        # 调整分配
        final = []

        # 添加文本结果
        for r in text_results[:target_text]:
            final.append(r)

        # 添加图像结果
        for r in image_results[:target_image]:
            final.append(r)

        # 如果不够，从另一模态补充
        remaining = total_k - len(final)
        if remaining > 0:
            # 从所有结果中补充
            for r in results:
                if r not in final:
                    final.append(r)
                    remaining -= 1
                    if remaining <= 0:
                        break

        # 按分数排序
        final.sort(key=lambda x: x.score, reverse=True)

        return final[:total_k]

    def retrieve_with_fixed_k(
        self,
        query: str,
        k_text: int = 2,
        k_image: int = 2,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        固定K值检索（用于基线对比）

        Args:
            query: 用户查询
            k_text: 文本检索数量
            k_image: 图像检索数量
            filter_doc_ids: 文档过滤

        Returns:
            检索结果列表
        """
        text_results = self.text_retriever.retrieve(
            query,
            k=k_text,
            filter_doc_ids=filter_doc_ids
        )

        image_results = []
        if k_image > 0 and self.image_retriever.index is not None:
            image_results = self.image_retriever.retrieve(
                query,
                k=k_image,
                filter_doc_ids=filter_doc_ids
            )

        # 简单合并
        all_results = text_results + image_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results

    def get_retrieval_statistics(self, amf_result: AMFResult) -> Dict[str, Any]:
        """获取检索统计信息"""
        results = amf_result.results

        text_count = sum(1 for r in results if r.modality == ModalityType.TEXT)
        image_count = sum(1 for r in results if r.modality == ModalityType.IMAGE)

        return {
            "total_results": len(results),
            "text_count": text_count,
            "image_count": image_count,
            "classification": {
                "preference": amf_result.classification.preference.value,
                "confidence": amf_result.classification.confidence,
                "reasoning": amf_result.classification.reasoning
            },
            "k_allocation": {
                "text": amf_result.k_allocation[0],
                "image": amf_result.k_allocation[1]
            },
            "reflection": amf_result.reflection_stats,
            "score_range": {
                "min": min(r.score for r in results) if results else 0,
                "max": max(r.score for r in results) if results else 0
            }
        }
