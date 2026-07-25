"""
可追溯答案生成器
生成带有引用的答案，确保可追溯性
"""

import re
import time
from typing import List, Optional, Dict, Any

from ..utils.logger import LoggerMixin
from ..utils.config import get_config, TAGConfig
from ..utils.llm_client import LLMClient
from ..utils.data_types import (
    RetrievalResult, TraceableAnswer, Citation, VerifiedCitation,
    SourceMetadata, ModalityType
)
from .citation_parser import CitationParser
from .citation_verifier import CitationVerifier


class TraceableAnswerGenerator(LoggerMixin):
    """
    可追溯答案生成器
    生成带有内联引用的答案，支持来源验证
    """

    GENERATION_PROMPT = """You are a technical expert answering questions based on provided documentation.

IMPORTANT RULES:
1. Only use information from the provided sources
2. Cite EVERY factual claim using [Source N] format immediately after the claim
3. If information is not in the sources, say "This information is not found in the provided documentation"
4. Do not hallucinate or add information not in sources
5. Be precise and technical in your response

Sources:
{formatted_sources}

Question: {question}

Provide a comprehensive answer with inline citations [Source N] for every factual statement.
Answer:"""

    GENERATION_WITH_IMAGES_PROMPT = """You are a technical expert answering questions based on provided documentation that includes both text and images.

IMPORTANT RULES:
1. Only use information from the provided sources (text and images)
2. Cite EVERY factual claim using [Source N] format
3. When referencing information from images, clearly indicate it (e.g., "As shown in [Source 2], the diagram indicates...")
4. If information is not in the sources, say "This information is not found in the provided documentation"
5. Do not hallucinate or add information not in sources

Sources:
{formatted_sources}

Question: {question}

Provide a comprehensive answer with inline citations [Source N] for every factual statement.
Answer:"""

    def __init__(self, config: Optional[TAGConfig] = None):
        self.config = config or get_config().tag
        self._llm_client = None
        self.citation_parser = CitationParser()
        self.citation_verifier = CitationVerifier() if self.config.verification_enabled else None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def generate(
        self,
        query: str,
        contexts: List[RetrievalResult]
    ) -> TraceableAnswer:
        """
        生成带引用的可追溯答案

        Args:
            query: 用户问题
            contexts: 检索到的上下文

        Returns:
            TraceableAnswer对象
        """
        start_time = time.time()

        # Step 1: 格式化来源
        formatted_sources = self._format_sources(contexts)

        # Step 2: 选择prompt
        has_images = any(c.modality == ModalityType.IMAGE for c in contexts)
        if has_images:
            prompt = self.GENERATION_WITH_IMAGES_PROMPT.format(
                formatted_sources=formatted_sources,
                question=query
            )
        else:
            prompt = self.GENERATION_PROMPT.format(
                formatted_sources=formatted_sources,
                question=query
            )

        # Step 3: 生成答案
        try:
            raw_answer = self.llm_client.generate(prompt)
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            raw_answer = "I apologize, but I was unable to generate an answer due to a technical error."

        # Step 4: 解析引用
        answer_text, citations = self.citation_parser.parse(raw_answer, contexts)

        # Step 5: 验证引用
        verified_citations = citations
        if self.citation_verifier and self.config.verification_enabled:
            verified_citations = self.citation_verifier.verify_batch(citations, contexts)

        # Step 6: 计算置信度
        confidence = self._compute_confidence(verified_citations, contexts)

        generation_time = time.time() - start_time

        return TraceableAnswer(
            query=query,
            answer=answer_text,
            citations=verified_citations,
            sources=contexts,
            confidence=confidence,
            generation_time=generation_time,
            metadata={
                "has_images": has_images,
                "num_sources": len(contexts),
                "num_citations": len(verified_citations),
                "verified_citations": sum(1 for c in verified_citations if c.verified)
            }
        )

    def _format_sources(self, contexts: List[RetrievalResult]) -> str:
        """格式化来源供LLM使用"""
        formatted = []

        for i, ctx in enumerate(contexts, 1):
            source_info = (
                f"[Source {i}] "
                f"Document: {ctx.source.document}, "
                f"Page: {ctx.source.page}, "
                f"Section: {ctx.source.section or 'N/A'}, "
                f"Type: {ctx.modality.value}"
            )

            if ctx.modality == ModalityType.IMAGE:
                content = f"[IMAGE] {ctx.content}" if ctx.content else "[IMAGE - No description available]"
            else:
                content = ctx.content

            formatted.append(f"{source_info}\n{content}\n")

        return "\n---\n".join(formatted)

    def _compute_confidence(
        self,
        citations: List[VerifiedCitation],
        contexts: List[RetrievalResult]
    ) -> float:
        """计算答案置信度"""
        if not citations:
            return 0.3  # 无引用时低置信度

        # 基于引用验证计算
        verified_count = sum(1 for c in citations if c.verified)
        verification_ratio = verified_count / len(citations) if citations else 0

        # 基于来源相关性
        avg_source_score = sum(c.score for c in contexts) / len(contexts) if contexts else 0

        # 综合置信度
        confidence = 0.5 * verification_ratio + 0.3 * avg_source_score + 0.2

        return min(confidence, 1.0)

    def generate_with_rejection(
        self,
        query: str,
        contexts: List[RetrievalResult],
        min_confidence: float = 0.5
    ) -> TraceableAnswer:
        """
        带拒绝机制的答案生成
        如果置信度过低，拒绝回答

        Args:
            query: 用户问题
            contexts: 检索到的上下文
            min_confidence: 最小置信度阈值

        Returns:
            TraceableAnswer对象
        """
        # 首先检查上下文质量
        if not contexts:
            return TraceableAnswer(
                query=query,
                answer="I cannot answer this question as no relevant documentation was found.",
                citations=[],
                sources=[],
                confidence=0.0,
                generation_time=0.0,
                metadata={"rejected": True, "reason": "no_context"}
            )

        # 检查上下文相关性
        avg_score = sum(c.score for c in contexts) / len(contexts)
        if avg_score < 0.3:
            return TraceableAnswer(
                query=query,
                answer="I cannot confidently answer this question as the retrieved documentation does not appear to be sufficiently relevant.",
                citations=[],
                sources=contexts,
                confidence=avg_score,
                generation_time=0.0,
                metadata={"rejected": True, "reason": "low_relevance"}
            )

        # 正常生成
        answer = self.generate(query, contexts)

        # 检查生成结果的置信度
        if answer.confidence < min_confidence:
            answer.metadata["low_confidence_warning"] = True
            answer.answer = (
                f"[Note: This answer has relatively low confidence ({answer.confidence:.2f})]\n\n"
                f"{answer.answer}"
            )

        return answer
