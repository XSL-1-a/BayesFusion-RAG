"""
章节检测模块
识别文档的层次结构：章节、小节、段落
"""

import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_types import Section
from .pdf_parser import ParsedDocument, ParsedPage


@dataclass
class HeadingPattern:
    """标题模式"""
    pattern: str
    level: int
    name: str
    case_sensitive: bool = True  # 默认大小写敏感


class SectionDetector(LoggerMixin):
    """
    章节检测器
    从PDF文档中识别章节结构
    """

    # 常见的标题模式
    DEFAULT_PATTERNS = [
        # 英文标题（大小写不敏感）
        HeadingPattern(r'^Chapter\s+(\d+)[.:]\s*(.+)$', 1, "chapter", case_sensitive=False),
        HeadingPattern(r'^Section\s+(\d+(?:\.\d+)?)[.:]\s*(.+)$', 2, "section", case_sensitive=False),
        # 技术手册特有格式：数字-数字 标题 (如 "4-2 REMOVAL AND INSTALLATION...")
        # 大小写敏感，要求全大写标题
        HeadingPattern(r'^(\d+-\d+)\s+([A-Z][A-Z\s\-]{15,})$', 2, "tech_manual_section"),
        # 更严格的编号标题：需要数字后跟全大写标题（不匹配步骤说明如"1 Loosen..."）
        HeadingPattern(r'^(\d+)\s+([A-Z][A-Z\s\-]{15,})$', 1, "numbered_h1"),
        HeadingPattern(r'^(\d+\.\d+)\s+([A-Z][A-Z\s\-]{15,})$', 2, "numbered_h2"),
        HeadingPattern(r'^(\d+\.\d+\.\d+)\s+([A-Z][A-Z\s\-]{10,})$', 3, "numbered_h3"),
        HeadingPattern(r'^([IVXLCDM]+)\.\s*([A-Z][A-Z\s]{15,})$', 1, "roman_h1"),

        # 中文标题
        HeadingPattern(r'^第([一二三四五六七八九十]+)章\s*(.+)$', 1, "zh_chapter"),
        HeadingPattern(r'^第([一二三四五六七八九十]+)节\s*(.+)$', 2, "zh_section"),
        HeadingPattern(r'^([一二三四五六七八九十]+)[、.]\s*(.{5,})$', 2, "zh_numbered"),

        # 全大写标题 - 更严格：至少20字符，且包含多个单词
        HeadingPattern(r'^([A-Z][A-Z\s]{20,60})$', 1, "uppercase_heading"),
    ]

    def __init__(
        self,
        patterns: Optional[List[HeadingPattern]] = None,
        min_section_length: int = 100
    ):
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.min_section_length = min_section_length

    def detect_sections(self, parsed_doc: ParsedDocument) -> List[Section]:
        """
        检测文档章节结构

        Args:
            parsed_doc: 解析后的文档

        Returns:
            Section列表
        """
        sections = []
        current_section = None
        section_idx = 0

        for page in parsed_doc.pages:
            lines = page.text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检测是否是标题
                heading_info = self._detect_heading(line)

                if heading_info:
                    # 保存之前的section
                    if current_section:
                        sections.append(current_section)

                    # 创建新section
                    level, title = heading_info
                    current_section = Section(
                        section_id=f"{parsed_doc.doc_id}_sec_{section_idx}",
                        doc_id=parsed_doc.doc_id,
                        title=title,
                        level=level,
                        start_page=page.page_num,
                        end_page=page.page_num,
                        content=""
                    )
                    section_idx += 1

                elif current_section:
                    # 添加内容到当前section
                    current_section.content += line + "\n"
                    current_section.end_page = page.page_num

        # 保存最后一个section
        if current_section:
            sections.append(current_section)

        # 生成章节摘要
        for section in sections:
            section.abstract = self._generate_abstract(section.content)

        self.logger.info(f"Detected {len(sections)} sections in {parsed_doc.name}")
        return sections

    def _detect_heading(self, line: str) -> Optional[Tuple[int, str]]:
        """
        检测行是否是标题

        Args:
            line: 文本行

        Returns:
            (level, title) 或 None
        """
        line = line.strip()

        # 检查长度限制
        if len(line) > 200 or len(line) < 2:
            return None

        for pattern in self.patterns:
            # 根据 case_sensitive 标志决定是否区分大小写
            flags = 0 if pattern.case_sensitive else re.IGNORECASE
            match = re.match(pattern.pattern, line, flags)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    title = groups[-1].strip()
                else:
                    title = line
                return (pattern.level, title)

        # 启发式检测：短行 + 特定格式
        if self._is_likely_heading(line):
            return (2, line)

        return None

    def _is_likely_heading(self, line: str) -> bool:
        """启发式判断是否可能是标题 - 更严格的规则避免误判"""
        # 排除明显不是标题的情况
        if len(line) < 15 or len(line) > 70:
            return False

        # 排除以标点结尾
        if line.endswith(('.', ',', ':', ';', '。', '，', ')', '）')):
            return False

        # 排除包含数字编号的步骤说明 (如 "1 Loosen two hose...")
        if line[0].isdigit() and ' ' in line[:5]:
            return False

        # 排除包含括号引用的行 (如 "item 14, Appx C")
        if '(' in line and ')' in line:
            return False

        # 排除包含连字符编号的行 (如 "TM 9-2350-311-34-1")
        if line.count('-') >= 3:
            return False

        # 只有全大写且包含多个单词的才可能是标题
        if line.isupper() and line.count(' ') >= 2 and len(line) >= 20:
            return True

        return False

    def _generate_abstract(self, content: str, max_length: int = 300) -> str:
        """
        生成章节摘要

        Args:
            content: 章节内容
            max_length: 最大长度

        Returns:
            摘要文本
        """
        if not content:
            return ""

        # 简单实现：取前N个字符
        content = content.strip()

        if len(content) <= max_length:
            return content

        # 在句子边界截断
        truncated = content[:max_length]
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('。'),
            truncated.rfind('\n')
        )

        if last_period > max_length // 2:
            return truncated[:last_period + 1].strip()

        return truncated.strip() + "..."

    def build_section_tree(self, sections: List[Section]) -> Dict:
        """
        构建章节层次树

        Args:
            sections: Section列表

        Returns:
            层次结构字典
        """
        tree = {"root": [], "children": {}}

        stack = []  # (level, section_id)

        for section in sections:
            node = {
                "section": section,
                "children": []
            }

            # 找到父节点
            while stack and stack[-1][0] >= section.level:
                stack.pop()

            if stack:
                parent_id = stack[-1][1]
                if parent_id not in tree["children"]:
                    tree["children"][parent_id] = []
                tree["children"][parent_id].append(node)
            else:
                tree["root"].append(node)

            stack.append((section.level, section.section_id))

        return tree

    def get_section_hierarchy_text(self, sections: List[Section]) -> str:
        """
        生成章节层次文本表示

        Args:
            sections: Section列表

        Returns:
            层次文本
        """
        lines = []
        for section in sections:
            indent = "  " * (section.level - 1)
            lines.append(f"{indent}{section.level}. {section.title}")
        return "\n".join(lines)
