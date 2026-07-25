"""
结构化输出格式化器
将答案格式化为各种输出格式
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..utils.data_types import TraceableAnswer, VerifiedCitation


class StructuredOutputFormatter(LoggerMixin):
    """
    结构化输出格式化器
    支持多种输出格式：JSON、Markdown、HTML
    """

    def format_json(
        self,
        answer: TraceableAnswer,
        include_sources: bool = True,
        pretty: bool = True
    ) -> str:
        """
        格式化为JSON

        Args:
            answer: TraceableAnswer对象
            include_sources: 是否包含完整来源
            pretty: 是否美化输出

        Returns:
            JSON字符串
        """
        output = {
            "query": answer.query,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "citations": [
                {
                    "id": c.citation_id,
                    "claim": c.claim,
                    "source": {
                        "document": c.source.document,
                        "page": c.source.page,
                        "section": c.source.section
                    },
                    "verified": c.verified,
                    "evidence": c.evidence
                }
                for c in answer.citations
            ],
            "metadata": {
                "generation_time": answer.generation_time,
                "num_sources": len(answer.sources),
                "num_citations": len(answer.citations),
                "timestamp": datetime.now().isoformat()
            }
        }

        if include_sources:
            output["sources"] = [
                {
                    "id": i + 1,
                    "document": s.source.document,
                    "page": s.source.page,
                    "section": s.source.section,
                    "modality": s.modality.value,
                    "content_preview": s.content[:500] + "..." if len(s.content) > 500 else s.content,
                    "score": s.score
                }
                for i, s in enumerate(answer.sources)
            ]

        if pretty:
            return json.dumps(output, indent=2, ensure_ascii=False)
        return json.dumps(output, ensure_ascii=False)

    def format_markdown(
        self,
        answer: TraceableAnswer,
        include_sources: bool = True
    ) -> str:
        """
        格式化为Markdown

        Args:
            answer: TraceableAnswer对象
            include_sources: 是否包含来源列表

        Returns:
            Markdown字符串
        """
        lines = []

        # 问题
        lines.append(f"## Question\n\n{answer.query}\n")

        # 答案
        lines.append(f"## Answer\n\n{answer.answer}\n")

        # 置信度
        confidence_emoji = "🟢" if answer.confidence > 0.7 else "🟡" if answer.confidence > 0.4 else "🔴"
        lines.append(f"\n**Confidence:** {confidence_emoji} {answer.confidence:.2%}\n")

        # 引用详情
        if answer.citations:
            lines.append("\n## Citation Details\n")
            for c in answer.citations:
                status = "✓" if c.verified else "?"
                lines.append(
                    f"- **[Source {c.citation_id}]** {status}\n"
                    f"  - Claim: \"{c.claim[:100]}...\"\n"
                    f"  - Source: {c.source.document}, Page {c.source.page}\n"
                )
                if c.evidence:
                    lines.append(f"  - Evidence: \"{c.evidence[:150]}...\"\n")

        # 来源列表
        if include_sources:
            lines.append("\n## Sources\n")
            for i, s in enumerate(answer.sources, 1):
                lines.append(
                    f"{i}. **{s.source.document}** (Page {s.source.page})\n"
                    f"   - Type: {s.modality.value}\n"
                    f"   - Relevance: {s.score:.2%}\n"
                )

        # 元数据
        lines.append(f"\n---\n*Generated in {answer.generation_time:.2f}s*\n")

        return "\n".join(lines)

    def format_html(
        self,
        answer: TraceableAnswer,
        include_sources: bool = True
    ) -> str:
        """
        格式化为HTML

        Args:
            answer: TraceableAnswer对象
            include_sources: 是否包含来源

        Returns:
            HTML字符串
        """
        # 构建引用工具提示
        citation_tooltips = {}
        for c in answer.citations:
            tooltip = f"Source: {c.source.document}, Page {c.source.page}"
            if c.evidence:
                tooltip += f"\nEvidence: {c.evidence[:100]}..."
            citation_tooltips[c.citation_id] = tooltip

        # 替换引用为带工具提示的链接
        import re
        answer_html = answer.answer

        def replace_citation(match):
            num = int(match.group(1))
            tooltip = citation_tooltips.get(num, "")
            verified = any(c.verified for c in answer.citations if c.citation_id == num)
            color = "#28a745" if verified else "#ffc107"
            return f'<sup><a href="#source-{num}" title="{tooltip}" style="color: {color};">[{num}]</a></sup>'

        answer_html = re.sub(r'\[Source\s*(\d+)\]', replace_citation, answer_html)

        # 构建完整HTML
        html_parts = [
            '<div class="traceable-answer">',
            f'<h3>Question</h3><p>{answer.query}</p>',
            f'<h3>Answer</h3><div class="answer-content">{answer_html}</div>',
        ]

        # 置信度
        conf_class = "high" if answer.confidence > 0.7 else "medium" if answer.confidence > 0.4 else "low"
        html_parts.append(
            f'<div class="confidence {conf_class}">'
            f'Confidence: {answer.confidence:.2%}</div>'
        )

        # 来源列表
        if include_sources:
            html_parts.append('<h3>Sources</h3><ol class="sources-list">')
            for i, s in enumerate(answer.sources, 1):
                verified_citations = [c for c in answer.citations if c.citation_id == i and c.verified]
                status_class = "verified" if verified_citations else "unverified"
                html_parts.append(
                    f'<li id="source-{i}" class="{status_class}">'
                    f'<strong>{s.source.document}</strong> (Page {s.source.page})<br>'
                    f'<small>Type: {s.modality.value}, Relevance: {s.score:.2%}</small>'
                    f'</li>'
                )
            html_parts.append('</ol>')

        html_parts.append('</div>')

        # 添加样式
        style = """
        <style>
        .traceable-answer { font-family: sans-serif; max-width: 800px; margin: 0 auto; }
        .answer-content { line-height: 1.6; }
        .confidence { padding: 8px; border-radius: 4px; margin: 10px 0; }
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
        .sources-list li { margin: 8px 0; padding: 8px; background: #f8f9fa; }
        .sources-list li.verified { border-left: 3px solid #28a745; }
        .sources-list li.unverified { border-left: 3px solid #ffc107; }
        </style>
        """

        return style + "\n".join(html_parts)

    def format_for_evaluation(
        self,
        answer: TraceableAnswer
    ) -> Dict[str, Any]:
        """
        格式化为评估所需的结构

        Args:
            answer: TraceableAnswer对象

        Returns:
            评估用字典
        """
        return {
            "query": answer.query,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "citations": {
                "total": len(answer.citations),
                "verified": sum(1 for c in answer.citations if c.verified),
                "precision": (
                    sum(1 for c in answer.citations if c.verified) / len(answer.citations)
                    if answer.citations else 0
                )
            },
            "sources": {
                "total": len(answer.sources),
                "cited": len(set(c.citation_id for c in answer.citations)),
                "avg_relevance": (
                    sum(s.score for s in answer.sources) / len(answer.sources)
                    if answer.sources else 0
                )
            },
            "generation_time": answer.generation_time,
            "metadata": answer.metadata
        }
