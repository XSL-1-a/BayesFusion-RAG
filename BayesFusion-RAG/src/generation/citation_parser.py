"""
引用解析器
从生成的答案中解析引用标记
"""

import re
from typing import List, Tuple, Optional

from ..utils.logger import LoggerMixin
from ..utils.data_types import (
    RetrievalResult, Citation, VerifiedCitation, SourceMetadata
)


class CitationParser(LoggerMixin):
    """
    引用解析器
    解析答案中的 [Source N] 引用标记
    """

    # 引用模式
    CITATION_PATTERN = r'\[Source\s*(\d+)\]'

    def __init__(self, citation_format: str = "[Source N]"):
        self.citation_format = citation_format
        self._pattern = re.compile(self.CITATION_PATTERN, re.IGNORECASE)

    def parse(
        self,
        answer_text: str,
        contexts: List[RetrievalResult]
    ) -> Tuple[str, List[VerifiedCitation]]:
        """
        解析答案中的引用

        Args:
            answer_text: 生成的答案文本
            contexts: 来源上下文

        Returns:
            (答案文本, 引用列表)
        """
        citations = []
        seen_claims = set()

        # 查找所有引用
        for match in self._pattern.finditer(answer_text):
            source_idx = int(match.group(1)) - 1  # 转换为0-indexed

            if 0 <= source_idx < len(contexts):
                # 提取引用前的声明文本
                claim = self._extract_claim(answer_text, match.start())

                # 避免重复
                claim_key = f"{source_idx}_{claim[:50]}"
                if claim_key in seen_claims:
                    continue
                seen_claims.add(claim_key)

                source = contexts[source_idx]
                citation = VerifiedCitation(
                    citation_id=source_idx + 1,
                    claim=claim,
                    source=source.source,
                    position=match.start(),
                    verified=False  # 待验证
                )
                citations.append(citation)

        self.logger.debug(f"Parsed {len(citations)} citations from answer")

        return answer_text, citations

    def _extract_claim(self, text: str, citation_pos: int) -> str:
        """
        提取引用前的声明文本

        Args:
            text: 完整文本
            citation_pos: 引用位置

        Returns:
            声明文本
        """
        # 向前查找到上一个句子结束或段落开始
        search_start = max(0, citation_pos - 500)
        preceding_text = text[search_start:citation_pos]

        # 尝试找到最近的句子
        # 从后向前查找句子分隔符
        separators = ['. ', '。', '\n', '; ', '；']

        last_sep_pos = -1
        for sep in separators:
            pos = preceding_text.rfind(sep)
            if pos > last_sep_pos:
                last_sep_pos = pos

        if last_sep_pos >= 0:
            claim = preceding_text[last_sep_pos + 1:].strip()
        else:
            claim = preceding_text.strip()

        # 清理
        claim = claim.strip()

        # 限制长度
        if len(claim) > 300:
            claim = claim[-300:]

        return claim

    def count_citations(self, answer_text: str) -> int:
        """统计引用数量"""
        return len(self._pattern.findall(answer_text))

    def get_cited_sources(
        self,
        answer_text: str,
        contexts: List[RetrievalResult]
    ) -> List[int]:
        """
        获取被引用的来源索引

        Args:
            answer_text: 答案文本
            contexts: 来源上下文

        Returns:
            被引用的来源索引列表（0-indexed）
        """
        cited = set()
        for match in self._pattern.finditer(answer_text):
            source_idx = int(match.group(1)) - 1
            if 0 <= source_idx < len(contexts):
                cited.add(source_idx)
        return list(cited)

    def get_uncited_sources(
        self,
        answer_text: str,
        contexts: List[RetrievalResult]
    ) -> List[int]:
        """获取未被引用的来源索引"""
        cited = set(self.get_cited_sources(answer_text, contexts))
        return [i for i in range(len(contexts)) if i not in cited]

    def normalize_citations(self, answer_text: str) -> str:
        """
        规范化引用格式

        Args:
            answer_text: 答案文本

        Returns:
            规范化后的文本
        """
        # 统一引用格式
        def replace_citation(match):
            num = match.group(1)
            return f"[Source {num}]"

        return self._pattern.sub(replace_citation, answer_text)
