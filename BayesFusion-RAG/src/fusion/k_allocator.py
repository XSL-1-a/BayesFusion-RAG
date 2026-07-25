"""
自适应K值分配器
根据查询特性动态分配文本和图像的检索数量
"""

from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_types import ModalityPreferenceType
from .query_classifier import ClassificationResult


@dataclass
class AllocationConfig:
    """分配配置"""
    text_dominant_ratio: Tuple[float, float] = (0.75, 0.25)
    image_dominant_ratio: Tuple[float, float] = (0.25, 0.75)
    balanced_ratio: Tuple[float, float] = (0.5, 0.5)
    min_k_per_modality: int = 1


class AdaptiveKAllocator(LoggerMixin):
    """
    自适应K值分配器
    根据查询分类结果动态调整文本和图像的检索数量
    """

    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()

    def allocate(
        self,
        classification: ClassificationResult,
        total_k: int = 4
    ) -> Tuple[int, int]:
        """
        根据分类结果分配K值

        Args:
            classification: 查询分类结果
            total_k: 总检索数量

        Returns:
            (k_text, k_image)
        """
        # 获取基础比例
        if classification.preference == ModalityPreferenceType.TEXT_DOMINANT:
            base_ratio = self.config.text_dominant_ratio
        elif classification.preference == ModalityPreferenceType.IMAGE_DOMINANT:
            base_ratio = self.config.image_dominant_ratio
        else:
            base_ratio = self.config.balanced_ratio

        # 根据置信度调整
        confidence = classification.confidence
        if confidence < 0.6:
            # 低置信度时趋向平衡
            adjusted_ratio = self._interpolate_ratio(
                base_ratio,
                (0.5, 0.5),
                1 - confidence  # 置信度越低，越接近平衡
            )
        else:
            adjusted_ratio = base_ratio

        # 计算分配
        k_text = max(
            self.config.min_k_per_modality,
            int(total_k * adjusted_ratio[0])
        )
        k_image = max(
            self.config.min_k_per_modality,
            total_k - k_text
        )

        # 确保总和等于total_k
        if k_text + k_image > total_k:
            if classification.preference == ModalityPreferenceType.TEXT_DOMINANT:
                k_image = total_k - k_text
            else:
                k_text = total_k - k_image

        self.logger.debug(
            f"K allocation: text={k_text}, image={k_image} "
            f"(preference={classification.preference.value}, confidence={confidence:.2f})"
        )

        return k_text, k_image

    def _interpolate_ratio(
        self,
        ratio1: Tuple[float, float],
        ratio2: Tuple[float, float],
        weight: float
    ) -> Tuple[float, float]:
        """在两个比例之间插值"""
        text_ratio = ratio1[0] * (1 - weight) + ratio2[0] * weight
        image_ratio = ratio1[1] * (1 - weight) + ratio2[1] * weight
        return (text_ratio, image_ratio)

    def allocate_with_feedback(
        self,
        classification: ClassificationResult,
        total_k: int,
        previous_relevance: Optional[dict] = None
    ) -> Tuple[int, int]:
        """
        带反馈的K值分配
        根据之前检索的相关性反馈调整分配

        Args:
            classification: 查询分类结果
            total_k: 总检索数量
            previous_relevance: 之前检索的相关性 {"text": float, "image": float}

        Returns:
            (k_text, k_image)
        """
        k_text, k_image = self.allocate(classification, total_k)

        if previous_relevance:
            text_rel = previous_relevance.get("text", 0.5)
            image_rel = previous_relevance.get("image", 0.5)

            # 根据相关性反馈调整
            if text_rel > image_rel + 0.2:
                # 文本更相关，增加文本配额
                k_text = min(total_k - 1, k_text + 1)
                k_image = total_k - k_text
            elif image_rel > text_rel + 0.2:
                # 图像更相关，增加图像配额
                k_image = min(total_k - 1, k_image + 1)
                k_text = total_k - k_image

        return k_text, k_image
