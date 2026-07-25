"""
文本分块模块
支持多种分块策略：固定大小、语义分块、递归分块
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logger import LoggerMixin
from ..utils.data_types import Chunk, ModalityType


@dataclass
class ChunkConfig:
    """分块配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separators: List[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "。", " ", ""]


class BaseChunker(ABC, LoggerMixin):
    """分块器基类"""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """分块"""
        pass

    def _create_chunk(
        self,
        content: str,
        chunk_idx: int,
        doc_id: str,
        section_id: Optional[str],
        page: int,
        start_char: int = 0
    ) -> Chunk:
        """创建Chunk对象"""
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            doc_id=doc_id,
            section_id=section_id,
            content=content,
            modality=ModalityType.TEXT,
            page=page,
            start_char=start_char,
            end_char=start_char + len(content)
        )


class FixedSizeChunker(BaseChunker):
    """固定大小分块器"""

    def chunk(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """固定大小分块"""
        chunks = []
        text = text.strip()

        if len(text) <= self.config.chunk_size:
            if len(text) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(
                    text, 0, doc_id, section_id, page, 0
                ))
            return chunks

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # 在句子边界处截断
            if end < len(text):
                # 尝试在句号处截断
                for sep in ['. ', '。', '\n', ' ']:
                    last_sep = text.rfind(sep, start + self.config.min_chunk_size, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text, chunk_idx, doc_id, section_id, page, start
                ))
                chunk_idx += 1

            start = end - self.config.chunk_overlap

        return chunks


class RecursiveChunker(BaseChunker):
    """递归分块器（LangChain风格）"""

    def chunk(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """递归分块"""
        chunks = []
        chunk_idx = [0]  # 使用列表以便在递归中修改

        def split_text(text: str, separators: List[str], start_pos: int = 0):
            if not separators:
                return [text]

            sep = separators[0]
            remaining_seps = separators[1:]

            if sep:
                parts = text.split(sep)
            else:
                # 最后的分隔符为空字符串时，按字符分割
                parts = list(text)

            result = []
            current_chunk = ""
            current_start = start_pos

            for i, part in enumerate(parts):
                test_chunk = current_chunk + (sep if current_chunk else "") + part

                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        if len(current_chunk) >= self.config.min_chunk_size:
                            result.append((current_chunk, current_start))
                        current_start += len(current_chunk) + len(sep)

                    if len(part) > self.config.chunk_size and remaining_seps:
                        # 递归处理
                        sub_results = split_text(part, remaining_seps, current_start)
                        result.extend(sub_results)
                        if sub_results:
                            current_start = sub_results[-1][1] + len(sub_results[-1][0]) + len(sep)
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                result.append((current_chunk, current_start))

            return result

        # 执行递归分块
        split_results = split_text(text.strip(), self.config.separators)

        for chunk_text, start_pos in split_results:
            chunks.append(self._create_chunk(
                chunk_text.strip(),
                chunk_idx[0],
                doc_id,
                section_id,
                page,
                start_pos
            ))
            chunk_idx[0] += 1

        return chunks


class SemanticChunker(BaseChunker):
    """语义分块器（基于嵌入相似度）"""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        similarity_threshold: float = 0.7
    ):
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    def _get_embedder(self):
        """延迟加载嵌入器"""
        if self._embedder is None:
            from ..utils.embedder import Embedder
            self._embedder = Embedder.get_text_embedder()
        return self._embedder

    def chunk(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """语义分块"""
        # 先按句子分割
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # 简化版：先使用固定分块，后续可扩展为真正的语义分块
        # 完整实现需要计算句子嵌入并基于相似度合并
        fixed_chunker = FixedSizeChunker(self.config)
        return fixed_chunker.chunk(text, doc_id, section_id, page)

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 简单的句子分割
        sentence_pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]


class TextChunker(LoggerMixin):
    """
    统一文本分块器
    支持多种分块策略
    """

    STRATEGIES = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker
    }

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.strategy = strategy
        self.config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        self._chunker = self._get_chunker()

    def _get_chunker(self) -> BaseChunker:
        """获取分块器实例"""
        chunker_class = self.STRATEGIES.get(self.strategy)
        if chunker_class is None:
            self.logger.warning(f"Unknown strategy: {self.strategy}, using recursive")
            chunker_class = RecursiveChunker
        return chunker_class(self.config)

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """分块文本"""
        return self._chunker.chunk(text, doc_id, section_id, page)

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        doc_id: str,
        section_id: Optional[str] = None
    ) -> List[Chunk]:
        """分块多页文本"""
        all_chunks = []
        for page_num, page_text in pages:
            chunks = self.chunk_text(page_text, doc_id, section_id, page_num)
            all_chunks.extend(chunks)

        # 重新编号chunk_id
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{doc_id}_chunk_{i}"

        return all_chunks

    def chunk_with_metadata(
        self,
        text: str,
        doc_id: str,
        metadata: dict,
        section_id: Optional[str] = None,
        page: int = 1
    ) -> List[Chunk]:
        """分块并添加元数据"""
        chunks = self.chunk_text(text, doc_id, section_id, page)
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks
