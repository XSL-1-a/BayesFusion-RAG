"""
文档处理器
整合PDF解析、分块、图像提取、章节检测、表格提取
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.data_types import Document, Section, Chunk, ImageElement, ModalityType

from .pdf_parser import PDFParser, ParsedDocument
from .text_chunker import TextChunker
from .image_extractor import ImageExtractor, ImageConfig
from .section_detector import SectionDetector

# 尝试导入 pdfplumber 用于表格提取
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


@dataclass
class ProcessedDocument:
    """处理后的完整文档"""
    document: Document
    sections: List[Section]
    chunks: List[Chunk]
    images: List[ImageElement]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "chunks": [c.to_dict() for c in self.chunks],
            "images": [i.to_dict() for i in self.images],
            "metadata": self.metadata
        }

    def save(self, path: str) -> None:
        """保存到JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProcessedDocument":
        """从JSON文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 需要实现反序列化逻辑
        raise NotImplementedError("Load from JSON not yet implemented")


class DocumentProcessor(LoggerMixin):
    """
    文档处理器
    完整的文档处理流水线
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive",
        extract_images: bool = True,
        generate_image_summaries: bool = True
    ):
        config = get_config()
        self.output_dir = Path(output_dir or config.paths.get("processed_dir", "./data/processed"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.pdf_parser = PDFParser(
            extract_images=extract_images,
            extract_tables=True,
            output_dir=str(self.output_dir)
        )

        self.text_chunker = TextChunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.image_extractor = ImageExtractor(
            config=ImageConfig(generate_summary=generate_image_summaries),
            output_dir=str(self.output_dir / "images")
        )

        self.section_detector = SectionDetector()

        self.logger.info("DocumentProcessor initialized")

    def process(self, pdf_path: str) -> ProcessedDocument:
        """
        处理单个PDF文档

        Args:
            pdf_path: PDF文件路径

        Returns:
            ProcessedDocument对象
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Processing document: {pdf_path.name}")

        # Step 1: 解析PDF
        parsed_doc = self.pdf_parser.parse(str(pdf_path))
        self.logger.info(f"Parsed {parsed_doc.num_pages} pages")

        # Step 2: 检测章节结构
        sections = self.section_detector.detect_sections(parsed_doc)
        self.logger.info(f"Detected {len(sections)} sections")

        # Step 3: 文本分块
        chunks = self._chunk_document(parsed_doc, sections)
        self.logger.info(f"Created {len(chunks)} chunks")

        # Step 3.5: 提取完整表格（修复跨页表格问题）
        table_chunks = self._extract_complete_tables(str(pdf_path), parsed_doc.doc_id)
        if table_chunks:
            chunks.extend(table_chunks)
            self.logger.info(f"Added {len(table_chunks)} complete table chunks")

        # Step 4: 提取图像
        images = self.image_extractor.extract_from_parsed_doc(parsed_doc)
        self.logger.info(f"Extracted {len(images)} images")

        # Step 5: 生成图像摘要
        if images:
            contexts = self._get_image_contexts(images, chunks)
            self.image_extractor.generate_summaries_batch(images, contexts)

        # Step 6: 创建Document对象
        document = Document(
            doc_id=parsed_doc.doc_id,
            name=parsed_doc.name,
            path=str(pdf_path),
            num_pages=parsed_doc.num_pages,
            summary=self._generate_doc_summary(parsed_doc, sections),
            sections=sections,
            metadata=parsed_doc.metadata
        )

        # Step 7: 关联章节到chunks
        self._assign_sections_to_chunks(chunks, sections)

        # 创建处理后的文档
        processed = ProcessedDocument(
            document=document,
            sections=sections,
            chunks=chunks,
            images=images,
            metadata={
                "num_pages": parsed_doc.num_pages,
                "num_sections": len(sections),
                "num_chunks": len(chunks),
                "num_images": len(images)
            }
        )

        # 保存处理结果
        output_path = self.output_dir / f"{parsed_doc.doc_id}.json"
        processed.save(str(output_path))
        self.logger.info(f"Saved processed document to {output_path}")

        return processed

    def process_directory(self, dir_path: str) -> List[ProcessedDocument]:
        """
        处理目录下所有PDF

        Args:
            dir_path: 目录路径

        Returns:
            ProcessedDocument列表
        """
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("**/*.pdf"))

        self.logger.info(f"Found {len(pdf_files)} PDF files")

        processed_docs = []
        for pdf_file in pdf_files:
            try:
                processed = self.process(str(pdf_file))
                processed_docs.append(processed)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {e}")

        return processed_docs

    def _chunk_document(
        self,
        parsed_doc: ParsedDocument,
        sections: List[Section]
    ) -> List[Chunk]:
        """分块文档"""
        all_chunks = []

        # 方法1：按页分块（简单但不考虑章节边界）
        # for page in parsed_doc.pages:
        #     chunks = self.text_chunker.chunk_text(
        #         page.text,
        #         parsed_doc.doc_id,
        #         page=page.page_num
        #     )
        #     all_chunks.extend(chunks)

        # 方法2：按章节分块（保留结构信息）
        if sections:
            for section in sections:
                if section.content and len(section.content) > 50:
                    chunks = self.text_chunker.chunk_text(
                        section.content,
                        parsed_doc.doc_id,
                        section_id=section.section_id,
                        page=section.start_page
                    )
                    # 更新chunk的元数据
                    for chunk in chunks:
                        chunk.metadata["section_title"] = section.title
                        chunk.metadata["section_level"] = section.level
                    all_chunks.extend(chunks)
        else:
            # 如果没有检测到章节，按页分块
            for page in parsed_doc.pages:
                chunks = self.text_chunker.chunk_text(
                    page.text,
                    parsed_doc.doc_id,
                    page=page.page_num
                )
                all_chunks.extend(chunks)

        # 重新编号
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{parsed_doc.doc_id}_chunk_{i}"

        return all_chunks

    def _generate_doc_summary(
        self,
        parsed_doc: ParsedDocument,
        sections: List[Section]
    ) -> str:
        """生成文档摘要"""
        # 简单实现：使用章节标题和首段
        summary_parts = []

        # 添加章节层次
        if sections:
            section_outline = self.section_detector.get_section_hierarchy_text(sections[:10])
            summary_parts.append(f"Document Structure:\n{section_outline}")

        # 添加首页内容摘要
        if parsed_doc.pages:
            first_page = parsed_doc.pages[0].text[:500]
            summary_parts.append(f"Overview:\n{first_page}")

        return "\n\n".join(summary_parts)

    def _get_image_contexts(
        self,
        images: List[ImageElement],
        chunks: List[Chunk]
    ) -> Dict[str, str]:
        """获取图像周围的文本上下文"""
        contexts = {}

        for image in images:
            # 找到同页的chunks
            page_chunks = [c for c in chunks if c.page == image.page]
            if page_chunks:
                context = " ".join(c.content[:200] for c in page_chunks[:2])
                contexts[image.image_id] = context

        return contexts

    def _assign_sections_to_chunks(
        self,
        chunks: List[Chunk],
        sections: List[Section]
    ) -> None:
        """为chunks分配所属章节"""
        # 构建页码到章节的映射
        page_to_section = {}
        for section in sections:
            for page in range(section.start_page, section.end_page + 1):
                if page not in page_to_section:
                    page_to_section[page] = []
                page_to_section[page].append(section)

        # 分配章节
        for chunk in chunks:
            if chunk.section_id is None and chunk.page in page_to_section:
                # 使用最具体的（最深层级的）章节
                page_sections = page_to_section[chunk.page]
                if page_sections:
                    deepest = max(page_sections, key=lambda s: s.level)
                    chunk.section_id = deepest.section_id
                    chunk.metadata["section_title"] = deepest.title

    def get_document_stats(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """获取文档统计信息"""
        return {
            "document_name": processed_doc.document.name,
            "num_pages": processed_doc.document.num_pages,
            "num_sections": len(processed_doc.sections),
            "num_chunks": len(processed_doc.chunks),
            "num_images": len(processed_doc.images),
            "avg_chunk_length": sum(len(c.content) for c in processed_doc.chunks) / max(1, len(processed_doc.chunks)),
            "sections_by_level": self._count_sections_by_level(processed_doc.sections)
        }

    def _count_sections_by_level(self, sections: List[Section]) -> Dict[int, int]:
        """统计各级章节数量"""
        counts = {}
        for section in sections:
            counts[section.level] = counts.get(section.level, 0) + 1
        return counts

    def _extract_complete_tables(self, pdf_path: str, doc_id: str) -> List[Chunk]:
        """
        提取完整的表格内容（解决跨页表格问题）

        使用 pdfplumber 直接提取页面文本，识别并合并跨页表格。

        Args:
            pdf_path: PDF文件路径
            doc_id: 文档ID

        Returns:
            包含完整表格的 Chunk 列表
        """
        if not HAS_PDFPLUMBER:
            self.logger.warning("pdfplumber not installed, skipping table extraction")
            return []

        table_chunks = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 1. 识别可能包含表格的页面区域（附录、表格标题等）
                table_regions = self._identify_table_regions(pdf)

                if not table_regions:
                    self.logger.info("No table regions identified")
                    return []

                self.logger.info(f"Found {len(table_regions)} table regions")

                # 2. 提取每个表格区域的完整内容
                for region in table_regions:
                    table_content = self._extract_table_region(pdf, region)
                    if table_content and len(table_content) > 100:
                        chunk = Chunk(
                            chunk_id=f"{doc_id}_complete_table_{region['name']}",
                            content=table_content,
                            doc_id=doc_id,
                            page=region['start_page'],
                            section_id=None,
                            modality=ModalityType.TABLE,
                            metadata={
                                "source": "complete_table_extraction",
                                "table_name": region['name'],
                                "page_range": f"{region['start_page']}-{region['end_page']}",
                                "is_table": True
                            }
                        )
                        table_chunks.append(chunk)
                        self.logger.info(f"Extracted table: {region['name']} ({len(table_content)} chars)")

        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")

        return table_chunks

    def _identify_table_regions(self, pdf) -> List[Dict[str, Any]]:
        """
        识别 PDF 中的表格区域

        通过扫描页面内容，识别包含表格的连续页面区域。
        """
        regions = []
        total_pages = len(pdf.pages)

        # 常见的附录/表格模式
        appendix_patterns = [
            (r'APPENDIX\s+([A-Z])', 'appendix'),
            (r'TABLE\s+([A-Z0-9-]+)', 'table'),
            (r'([A-Z])-\d+\s+.*(?:LIST|TABLE|PARTS)', 'list'),
        ]

        current_region = None

        for page_num in range(total_pages):
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            first_lines = text[:500]

            # 检查是否是新的表格区域开始
            for pattern, region_type in appendix_patterns:
                match = re.search(pattern, first_lines, re.IGNORECASE)
                if match:
                    # 保存之前的区域
                    if current_region and current_region['end_page'] == page_num:
                        regions.append(current_region)

                    region_name = match.group(1) if match.groups() else f"table_{page_num}"

                    # 检查是否是续页
                    if 'CONTINUED' in first_lines.upper() and current_region:
                        current_region['end_page'] = page_num + 1
                    else:
                        # 新区域
                        if current_region:
                            regions.append(current_region)
                        current_region = {
                            'name': f"{region_type}_{region_name}",
                            'start_page': page_num + 1,
                            'end_page': page_num + 1,
                            'type': region_type
                        }
                    break

            # 检查是否是续页
            if current_region and 'CONTINUED' in first_lines.upper():
                current_region['end_page'] = page_num + 1

        # 保存最后一个区域
        if current_region:
            regions.append(current_region)

        # 合并相邻的同类型区域
        merged_regions = []
        for region in regions:
            if merged_regions and merged_regions[-1]['name'] == region['name']:
                merged_regions[-1]['end_page'] = region['end_page']
            else:
                merged_regions.append(region)

        return merged_regions

    def _extract_table_region(self, pdf, region: Dict[str, Any]) -> str:
        """
        提取指定区域的完整表格内容
        """
        content_parts = []

        # 添加表格标题
        content_parts.append(f"{region['name'].upper()} (第 {region['start_page']}-{region['end_page']} 页)\n")

        for page_num in range(region['start_page'] - 1, region['end_page']):
            if page_num >= len(pdf.pages):
                break

            page = pdf.pages[page_num]
            text = page.extract_text() or ""

            if text:
                content_parts.append(f"--- 第 {page_num + 1} 页 ---\n{text}\n")

        return "\n".join(content_parts)
