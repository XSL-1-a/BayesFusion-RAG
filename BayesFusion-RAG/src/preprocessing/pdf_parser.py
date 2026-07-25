"""
PDF解析模块
提取PDF中的文本、图像、表格和结构信息
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

from ..utils.logger import LoggerMixin
from ..utils.data_types import Document, Section, Chunk, ImageElement, ModalityType


@dataclass
class ParsedPage:
    """解析后的页面"""
    page_num: int
    text: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    layout_info: Optional[Dict[str, Any]] = None


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    name: str
    path: str
    num_pages: int
    pages: List[ParsedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFParser(LoggerMixin):
    """
    PDF解析器
    支持多种PDF解析库，提取文本、图像和表格
    """

    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        ocr_enabled: bool = True,
        dpi: int = 300,
        output_dir: Optional[str] = None
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.ocr_enabled = ocr_enabled
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path("./data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._parser_backend = self._detect_backend()
        self.logger.info(f"Using PDF parser backend: {self._parser_backend}")

    def _detect_backend(self) -> str:
        """检测可用的PDF解析后端"""
        # 优先级：pymupdf > pdfplumber > pypdf
        try:
            import fitz  # PyMuPDF
            return "pymupdf"
        except ImportError:
            pass

        try:
            import pdfplumber
            return "pdfplumber"
        except ImportError:
            pass

        try:
            import pypdf
            return "pypdf"
        except ImportError:
            pass

        self.logger.warning("No PDF parser found. Install: pip install pymupdf pdfplumber")
        return "none"

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        解析PDF文档

        Args:
            pdf_path: PDF文件路径

        Returns:
            ParsedDocument对象
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # 生成文档ID
        doc_id = self._generate_doc_id(pdf_path)

        # 创建输出目录
        doc_output_dir = self.output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Parsing PDF: {pdf_path.name}")

        if self._parser_backend == "pymupdf":
            return self._parse_with_pymupdf(pdf_path, doc_id, doc_output_dir)
        elif self._parser_backend == "pdfplumber":
            return self._parse_with_pdfplumber(pdf_path, doc_id, doc_output_dir)
        elif self._parser_backend == "pypdf":
            return self._parse_with_pypdf(pdf_path, doc_id, doc_output_dir)
        else:
            raise RuntimeError("No PDF parser backend available")

    def _generate_doc_id(self, pdf_path: Path) -> str:
        """生成文档ID"""
        # 使用文件名和内容哈希
        with open(pdf_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()[:10000]).hexdigest()[:8]
        clean_name = re.sub(r'[^\w\-_]', '_', pdf_path.stem)
        return f"{clean_name}_{content_hash}"

    def _parse_with_pymupdf(
        self,
        pdf_path: Path,
        doc_id: str,
        output_dir: Path
    ) -> ParsedDocument:
        """使用PyMuPDF解析"""
        import fitz

        doc = fitz.open(pdf_path)
        pages = []
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 提取文本
            text = page.get_text("text")

            # 提取图像
            images = []
            if self.extract_images:
                image_list = page.get_images()
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # 保存图像
                        image_filename = f"page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                        image_path = image_dir / image_filename
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        images.append({
                            "image_id": f"{doc_id}_p{page_num + 1}_i{img_idx + 1}",
                            "path": str(image_path),
                            "page": page_num + 1,
                            "bbox": None  # PyMuPDF需要额外处理获取bbox
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image: {e}")

            # 提取表格（简化版）
            tables = []
            if self.extract_tables:
                try:
                    table_finder = page.find_tables()
                    for tbl_idx, table in enumerate(table_finder):
                        table_data = table.extract()
                        tables.append({
                            "table_id": f"{doc_id}_p{page_num + 1}_t{tbl_idx + 1}",
                            "page": page_num + 1,
                            "data": table_data,
                            "bbox": list(table.bbox)
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to extract tables: {e}")

            pages.append(ParsedPage(
                page_num=page_num + 1,
                text=text,
                images=images,
                tables=tables
            ))

        doc.close()

        return ParsedDocument(
            doc_id=doc_id,
            name=pdf_path.name,
            path=str(pdf_path),
            num_pages=len(pages),
            pages=pages,
            metadata={"parser": "pymupdf"}
        )

    def _parse_with_pdfplumber(
        self,
        pdf_path: Path,
        doc_id: str,
        output_dir: Path
    ) -> ParsedDocument:
        """使用pdfplumber解析"""
        import pdfplumber
        from PIL import Image
        import io

        pages = []
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 提取文本
                text = page.extract_text() or ""

                # 提取图像
                images = []
                if self.extract_images:
                    for img_idx, img in enumerate(page.images):
                        try:
                            # pdfplumber图像信息
                            image_filename = f"page{page_num + 1}_img{img_idx + 1}.png"
                            image_path = image_dir / image_filename

                            # 裁剪页面图像
                            x0, y0, x1, y1 = img['x0'], img['top'], img['x1'], img['bottom']
                            cropped = page.crop((x0, y0, x1, y1))
                            pil_image = cropped.to_image(resolution=self.dpi)
                            pil_image.save(str(image_path))

                            images.append({
                                "image_id": f"{doc_id}_p{page_num + 1}_i{img_idx + 1}",
                                "path": str(image_path),
                                "page": page_num + 1,
                                "bbox": [x0, y0, x1, y1]
                            })
                        except Exception as e:
                            self.logger.warning(f"Failed to extract image: {e}")

                # 提取表格
                tables = []
                if self.extract_tables:
                    for tbl_idx, table in enumerate(page.extract_tables()):
                        if table:
                            tables.append({
                                "table_id": f"{doc_id}_p{page_num + 1}_t{tbl_idx + 1}",
                                "page": page_num + 1,
                                "data": table,
                                "bbox": None
                            })

                pages.append(ParsedPage(
                    page_num=page_num + 1,
                    text=text,
                    images=images,
                    tables=tables
                ))

        return ParsedDocument(
            doc_id=doc_id,
            name=pdf_path.name,
            path=str(pdf_path),
            num_pages=len(pages),
            pages=pages,
            metadata={"parser": "pdfplumber"}
        )

    def _parse_with_pypdf(
        self,
        pdf_path: Path,
        doc_id: str,
        output_dir: Path
    ) -> ParsedDocument:
        """使用pypdf解析（基础版，仅文本）"""
        from pypdf import PdfReader

        pages = []
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(ParsedPage(
                page_num=page_num + 1,
                text=text,
                images=[],
                tables=[]
            ))

        return ParsedDocument(
            doc_id=doc_id,
            name=pdf_path.name,
            path=str(pdf_path),
            num_pages=len(pages),
            pages=pages,
            metadata={"parser": "pypdf"}
        )

    def parse_directory(self, dir_path: str) -> List[ParsedDocument]:
        """解析目录下所有PDF"""
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("**/*.pdf"))

        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.parse(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                self.logger.error(f"Failed to parse {pdf_file}: {e}")

        self.logger.info(f"Parsed {len(documents)} PDF documents")
        return documents

    def get_full_text(self, parsed_doc: ParsedDocument) -> str:
        """获取文档完整文本"""
        return "\n\n".join(page.text for page in parsed_doc.pages)

    def get_all_images(self, parsed_doc: ParsedDocument) -> List[Dict[str, Any]]:
        """获取文档所有图像"""
        images = []
        for page in parsed_doc.pages:
            images.extend(page.images)
        return images

    def get_all_tables(self, parsed_doc: ParsedDocument) -> List[Dict[str, Any]]:
        """获取文档所有表格"""
        tables = []
        for page in parsed_doc.pages:
            tables.extend(page.tables)
        return tables
