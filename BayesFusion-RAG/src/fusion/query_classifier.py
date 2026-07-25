"""
查询模态偏好分类器
分析查询并判断最适合的模态组合
"""

from typing import Optional
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.llm_client import LLMClient
from ..utils.config import get_config
from ..utils.data_types import ModalityPreference, ModalityPreferenceType


@dataclass
class ClassificationResult:
    """分类结果"""
    preference: ModalityPreferenceType
    confidence: float
    reasoning: str
    text_ratio: float
    image_ratio: float


class QueryModalityClassifier(LoggerMixin):
    """
    查询模态偏好分类器
    分析用户查询，判断应该更多依赖文本还是图像
    """

    CLASSIFICATION_PROMPT = """Analyze the following query and determine which modality (text or image) is more likely to contain the answer.

Query: {query}

Consider these guidelines:
- TEXT_DOMINANT: Queries about definitions, parameters, specifications, procedures, standards
  Examples: "What is the operating temperature range?", "Define the calibration procedure"

- IMAGE_DOMINANT: Queries about diagrams, schematics, visual layouts, component positions, physical appearance
  Examples: "Where is the sensor located?", "Show the circuit diagram", "What does the interface look like?"

- BALANCED: Queries that require both text explanation and visual reference
  Examples: "How to install the component?" (needs procedure text + diagram), "What are the connector pins?" (needs specs + pinout diagram)

Output a JSON object:
{{
    "modality_preference": "TEXT_DOMINANT" or "IMAGE_DOMINANT" or "BALANCED",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this modality is preferred",
    "text_relevance": 0.0-1.0,
    "image_relevance": 0.0-1.0
}}

Be decisive. Even if both modalities might help, choose the one more likely to contain the primary answer."""

    # 关键词启发式规则
    TEXT_KEYWORDS = [
        "define", "what is", "specification", "parameter", "value", "range",
        "procedure", "step", "standard", "requirement", "limit", "tolerance",
        "meaning", "description", "explanation", "list", "enumerate"
    ]

    IMAGE_KEYWORDS = [
        "diagram", "schematic", "layout", "position", "location", "where",
        "look like", "appearance", "visual", "image", "picture", "show",
        "circuit", "wiring", "connector", "pinout", "assembly", "exploded"
    ]

    BALANCED_KEYWORDS = [
        "how to", "install", "maintain", "troubleshoot", "connect",
        "interface", "setup", "configure", "assemble"
    ]

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def classify(self, query: str) -> ClassificationResult:
        """
        分类查询的模态偏好

        Args:
            query: 用户查询

        Returns:
            ClassificationResult对象
        """
        # 首先尝试启发式分类
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result.confidence >= self.confidence_threshold:
            self.logger.debug(f"Heuristic classification: {heuristic_result.preference}")
            return heuristic_result

        # 使用LLM进行更精确的分类
        return self._llm_classify(query)

    def _heuristic_classify(self, query: str) -> ClassificationResult:
        """基于关键词的启发式分类"""
        query_lower = query.lower()

        text_score = sum(1 for kw in self.TEXT_KEYWORDS if kw in query_lower)
        image_score = sum(1 for kw in self.IMAGE_KEYWORDS if kw in query_lower)
        balanced_score = sum(1 for kw in self.BALANCED_KEYWORDS if kw in query_lower)

        total = text_score + image_score + balanced_score + 1  # +1 避免除零

        if balanced_score > 0 and balanced_score >= max(text_score, image_score):
            preference = ModalityPreferenceType.BALANCED
            confidence = 0.6 + 0.1 * balanced_score
            text_ratio = 0.5
            image_ratio = 0.5
        elif text_score > image_score:
            preference = ModalityPreferenceType.TEXT_DOMINANT
            confidence = 0.5 + 0.1 * (text_score - image_score)
            text_ratio = 0.75
            image_ratio = 0.25
        elif image_score > text_score:
            preference = ModalityPreferenceType.IMAGE_DOMINANT
            confidence = 0.5 + 0.1 * (image_score - text_score)
            text_ratio = 0.25
            image_ratio = 0.75
        else:
            preference = ModalityPreferenceType.BALANCED
            confidence = 0.5
            text_ratio = 0.5
            image_ratio = 0.5

        return ClassificationResult(
            preference=preference,
            confidence=min(confidence, 1.0),
            reasoning=f"Heuristic: text_score={text_score}, image_score={image_score}, balanced_score={balanced_score}",
            text_ratio=text_ratio,
            image_ratio=image_ratio
        )

    def _llm_classify(self, query: str) -> ClassificationResult:
        """使用LLM分类"""
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = self.llm_client.generate_structured(prompt)

            preference_str = response.get("modality_preference", "BALANCED")
            try:
                preference = ModalityPreferenceType[preference_str]
            except KeyError:
                preference = ModalityPreferenceType.BALANCED

            confidence = response.get("confidence", 0.5)
            reasoning = response.get("reasoning", "")

            # 计算比例
            text_relevance = response.get("text_relevance", 0.5)
            image_relevance = response.get("image_relevance", 0.5)

            total = text_relevance + image_relevance
            if total > 0:
                text_ratio = text_relevance / total
                image_ratio = image_relevance / total
            else:
                text_ratio = 0.5
                image_ratio = 0.5

            return ClassificationResult(
                preference=preference,
                confidence=confidence,
                reasoning=reasoning,
                text_ratio=text_ratio,
                image_ratio=image_ratio
            )

        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
            # 回退到启发式
            return self._heuristic_classify(query)

    def get_recommended_k_allocation(
        self,
        classification: ClassificationResult,
        total_k: int = 4
    ) -> tuple:
        """
        根据分类结果推荐k值分配

        Args:
            classification: 分类结果
            total_k: 总检索数量

        Returns:
            (k_text, k_image)
        """
        k_text = max(1, int(total_k * classification.text_ratio))
        k_image = total_k - k_text

        return k_text, k_image
