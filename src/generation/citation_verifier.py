"""
引用验证器
验证引用的准确性和支持度
"""

from typing import List, Optional, Dict, Any

from ..utils.logger import LoggerMixin
from ..utils.llm_client import LLMClient
from ..utils.data_types import RetrievalResult, Citation, VerifiedCitation


class CitationVerifier(LoggerMixin):
    """
    引用验证器
    验证答案中的引用是否被来源文本支持
    """

    VERIFICATION_PROMPT = """Verify if the following claim is supported by the source text.

Claim made in the answer:
"{claim}"

Source text:
"{source}"

Evaluate:
1. Does the source text explicitly support this claim?
2. Is the claim a reasonable inference from the source?
3. Is there any contradiction between the claim and source?

Output a JSON object:
{{
    "supported": true or false,
    "support_level": "explicit" or "implicit" or "partial" or "none",
    "evidence": "exact quote from source that supports the claim (if supported, otherwise empty string)",
    "reason": "brief explanation of your judgment"
}}

Be strict: only mark as supported if the source clearly supports the claim."""

    def __init__(self, method: str = "llm"):
        """
        初始化验证器

        Args:
            method: 验证方法 ("llm", "nli", "rule")
        """
        self.method = method
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def verify(
        self,
        citation: VerifiedCitation,
        source_content: str
    ) -> VerifiedCitation:
        """
        验证单个引用

        Args:
            citation: 引用对象
            source_content: 来源内容

        Returns:
            验证后的引用
        """
        if self.method == "llm":
            return self._verify_with_llm(citation, source_content)
        elif self.method == "rule":
            return self._verify_with_rules(citation, source_content)
        else:
            # 默认使用规则
            return self._verify_with_rules(citation, source_content)

    def verify_batch(
        self,
        citations: List[VerifiedCitation],
        contexts: List[RetrievalResult]
    ) -> List[VerifiedCitation]:
        """
        批量验证引用

        Args:
            citations: 引用列表
            contexts: 来源上下文列表

        Returns:
            验证后的引用列表
        """
        verified = []

        for citation in citations:
            # 获取对应的来源内容
            source_idx = citation.citation_id - 1
            if 0 <= source_idx < len(contexts):
                source_content = contexts[source_idx].content
                verified_citation = self.verify(citation, source_content)
            else:
                # 无效引用
                citation.verified = False
                citation.verification_reason = "Invalid source reference"
                verified_citation = citation

            verified.append(verified_citation)

        self.logger.debug(
            f"Verified {len(verified)} citations, "
            f"{sum(1 for c in verified if c.verified)} supported"
        )

        return verified

    def _verify_with_llm(
        self,
        citation: VerifiedCitation,
        source_content: str
    ) -> VerifiedCitation:
        """使用LLM验证"""
        prompt = self.VERIFICATION_PROMPT.format(
            claim=citation.claim,
            source=source_content[:2000]
        )

        try:
            response = self.llm_client.generate_structured(prompt)

            citation.verified = response.get("supported", False)
            citation.evidence = response.get("evidence", "")
            citation.verification_reason = response.get("reason", "")

            # 添加支持级别到元数据
            if not hasattr(citation, 'metadata'):
                citation.metadata = {}

        except Exception as e:
            self.logger.warning(f"LLM verification failed: {e}")
            # 回退到规则验证
            return self._verify_with_rules(citation, source_content)

        return citation

    def _verify_with_rules(
        self,
        citation: VerifiedCitation,
        source_content: str
    ) -> VerifiedCitation:
        """使用规则验证"""
        claim_lower = citation.claim.lower()
        source_lower = source_content.lower()

        # 规则1：关键词重叠
        claim_words = set(claim_lower.split())
        source_words = set(source_lower.split())

        # 移除常见停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those'}

        claim_words = claim_words - stop_words
        source_words = source_words - stop_words

        if not claim_words:
            citation.verified = False
            citation.verification_reason = "Empty claim"
            return citation

        overlap = claim_words & source_words
        overlap_ratio = len(overlap) / len(claim_words)

        # 规则2：检查数字是否匹配
        import re
        claim_numbers = set(re.findall(r'\d+\.?\d*', citation.claim))
        source_numbers = set(re.findall(r'\d+\.?\d*', source_content))
        numbers_match = claim_numbers <= source_numbers if claim_numbers else True

        # 判断
        if overlap_ratio >= 0.5 and numbers_match:
            citation.verified = True
            citation.verification_reason = f"Keyword overlap: {overlap_ratio:.2%}"
            # 尝试找到证据
            citation.evidence = self._find_evidence(citation.claim, source_content)
        else:
            citation.verified = False
            citation.verification_reason = f"Low overlap: {overlap_ratio:.2%}"

        return citation

    def _find_evidence(self, claim: str, source: str, window: int = 200) -> str:
        """在来源中找到支持声明的证据片段"""
        claim_words = claim.lower().split()
        source_lower = source.lower()

        best_match = ""
        best_score = 0

        # 滑动窗口查找最佳匹配
        words = source.split()
        for i in range(0, len(words) - 20, 10):
            window_text = " ".join(words[i:i + 30])
            window_lower = window_text.lower()

            score = sum(1 for w in claim_words if w in window_lower)
            if score > best_score:
                best_score = score
                best_match = window_text

        return best_match[:window]

    def compute_citation_metrics(
        self,
        citations: List[VerifiedCitation]
    ) -> Dict[str, float]:
        """
        计算引用指标

        Args:
            citations: 验证后的引用列表

        Returns:
            指标字典
        """
        if not citations:
            return {
                "citation_count": 0,
                "verified_count": 0,
                "precision": 0.0,
                "has_evidence_ratio": 0.0
            }

        verified_count = sum(1 for c in citations if c.verified)
        has_evidence = sum(1 for c in citations if c.evidence)

        return {
            "citation_count": len(citations),
            "verified_count": verified_count,
            "precision": verified_count / len(citations),
            "has_evidence_ratio": has_evidence / len(citations)
        }
