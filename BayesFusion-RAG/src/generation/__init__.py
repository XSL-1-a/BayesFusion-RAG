"""
可追溯答案生成模块 (TAG - Traceable Answer Generation)
- 带引用的答案生成
- 引用解析与验证
- 结构化输出
- 幻觉检测
"""

from .generator import TraceableAnswerGenerator
from .citation_parser import CitationParser
from .citation_verifier import CitationVerifier
from .output_formatter import StructuredOutputFormatter

__all__ = [
    "TraceableAnswerGenerator",
    "CitationParser",
    "CitationVerifier",
    "StructuredOutputFormatter"
]
