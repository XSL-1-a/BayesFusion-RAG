"""
图像提取与处理模块
从PDF中提取图像，生成图像摘要
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_types import ImageElement, ModalityType, SourceMetadata


@dataclass
class ImageConfig:
    """图像处理配置"""
    max_size: tuple = (1024, 1024)
    min_size: tuple = (50, 50)
    formats: List[str] = None
    generate_summary: bool = True
    summary_max_length: int = 300

    def __post_init__(self):
        if self.formats is None:
            self.formats = ["png", "jpg", "jpeg", "gif", "webp"]


class ImageExtractor(LoggerMixin):
    """
    图像提取器
    从解析后的PDF中提取图像，生成摘要
    """

    def __init__(
        self,
        config: Optional[ImageConfig] = None,
        output_dir: Optional[str] = None
    ):
        self.config = config or ImageConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("./data/processed/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._llm_client = None

    def _get_llm_client(self):
        """延迟加载LLM客户端"""
        if self._llm_client is None:
            from ..utils.llm_client import LLMClient
            from ..utils.config import get_config
            self._llm_client = LLMClient.get_client(get_config().multimodal_llm)
        return self._llm_client

    def extract_from_parsed_doc(
        self,
        parsed_doc: "ParsedDocument",
        doc_output_dir: Optional[Path] = None
    ) -> List[ImageElement]:
        """
        从解析后的文档提取图像

        Args:
            parsed_doc: ParsedDocument对象
            doc_output_dir: 文档输出目录

        Returns:
            ImageElement列表
        """
        images = []

        for page in parsed_doc.pages:
            for img_info in page.images:
                try:
                    image_elem = self._process_image(
                        img_info,
                        parsed_doc.doc_id,
                        page.page_num
                    )
                    if image_elem:
                        images.append(image_elem)
                except Exception as e:
                    self.logger.warning(f"Failed to process image: {e}")

        self.logger.info(f"Extracted {len(images)} images from {parsed_doc.name}")
        return images

    def _process_image(
        self,
        img_info: Dict[str, Any],
        doc_id: str,
        page_num: int
    ) -> Optional[ImageElement]:
        """处理单个图像"""
        image_path = Path(img_info.get("path", ""))

        if not image_path.exists():
            return None

        # 检查图像大小
        if not self._check_image_size(image_path):
            self.logger.debug(f"Image too small or too large: {image_path}")
            return None

        # 确定图像类型
        image_type = self._detect_image_type(image_path)

        # 创建ImageElement
        image_elem = ImageElement(
            image_id=img_info.get("image_id", f"{doc_id}_img_{page_num}"),
            doc_id=doc_id,
            section_id=img_info.get("section_id"),
            page=page_num,
            image_path=str(image_path),
            image_type=image_type,
            caption=img_info.get("caption"),
            bounding_box=img_info.get("bbox"),
            metadata={}
        )

        return image_elem

    def _check_image_size(self, image_path: Path) -> bool:
        """检查图像大小是否在范围内"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size

                if width < self.config.min_size[0] or height < self.config.min_size[1]:
                    return False

                if width > self.config.max_size[0] * 2 or height > self.config.max_size[1] * 2:
                    # 图像过大，需要缩放
                    self._resize_image(image_path)

                return True
        except Exception as e:
            self.logger.warning(f"Failed to check image size: {e}")
            return False

    def _resize_image(self, image_path: Path) -> None:
        """缩放图像"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.thumbnail(self.config.max_size, Image.Resampling.LANCZOS)
                img.save(image_path, quality=95)
        except Exception as e:
            self.logger.warning(f"Failed to resize image: {e}")

    def _detect_image_type(self, image_path: Path) -> str:
        """
        检测图像类型
        Returns: diagram, photo, chart, table_image, unknown
        """
        # 简单的启发式判断
        # 完整实现可以使用图像分类模型
        filename = image_path.name.lower()

        if any(kw in filename for kw in ["diagram", "schematic", "circuit"]):
            return "diagram"
        elif any(kw in filename for kw in ["chart", "graph", "plot"]):
            return "chart"
        elif any(kw in filename for kw in ["table"]):
            return "table_image"
        elif any(kw in filename for kw in ["photo", "picture"]):
            return "photo"
        else:
            return "unknown"

    def generate_summary(
        self,
        image_elem: ImageElement,
        context: Optional[str] = None
    ) -> str:
        """
        生成图像摘要

        Args:
            image_elem: ImageElement对象
            context: 周围文本上下文

        Returns:
            图像摘要文本
        """
        if not self.config.generate_summary:
            return ""

        try:
            llm = self._get_llm_client()

            prompt = f"""Analyze this technical image and provide a concise summary.

Image Type: {image_elem.image_type}
{"Context from surrounding text: " + context[:500] if context else ""}

Describe:
1. What the image shows (components, diagrams, charts, etc.)
2. Key information visible in the image
3. Any text, labels, or values shown

Keep the summary under {self.config.summary_max_length} characters.
Be specific and technical."""

            summary = llm.generate_with_image(prompt, image_elem.image_path)
            image_elem.summary = summary[:self.config.summary_max_length]

            return image_elem.summary

        except Exception as e:
            self.logger.warning(f"Failed to generate image summary: {e}")
            return ""

    def generate_summaries_batch(
        self,
        images: List[ImageElement],
        contexts: Optional[Dict[str, str]] = None
    ) -> List[ImageElement]:
        """
        批量生成图像摘要

        Args:
            images: ImageElement列表
            contexts: 图像ID到上下文的映射

        Returns:
            更新后的ImageElement列表
        """
        contexts = contexts or {}

        for image in images:
            context = contexts.get(image.image_id, "")
            self.generate_summary(image, context)

        return images

    def extract_image_text_ocr(self, image_path: Union[str, Path]) -> str:
        """
        使用OCR提取图像中的文本

        Args:
            image_path: 图像路径

        Returns:
            提取的文本
        """
        try:
            import pytesseract
            from PIL import Image

            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img)
                return text.strip()

        except ImportError:
            self.logger.warning("pytesseract not installed. Run: pip install pytesseract")
            return ""
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return ""
