"""
层次化索引构建模块
构建三级索引：文档级 -> 章节级 -> 元素级
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils.logger import LoggerMixin
from ..utils.config import get_config, HMRConfig
from ..utils.embedder import Embedder, batch_cosine_similarity
from ..utils.data_types import Document, Section, Chunk, ImageElement


@dataclass
class IndexItem:
    """索引项"""
    item_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class VectorIndex(LoggerMixin):
    """
    向量索引
    支持向量存储和相似度检索
    """

    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.items: List[IndexItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def add(self, item: IndexItem) -> None:
        """添加索引项"""
        self._id_to_idx[item.item_id] = len(self.items)
        self.items.append(item)
        self.embeddings = None  # 标记需要重建

    def add_batch(self, items: List[IndexItem]) -> None:
        """批量添加索引项"""
        for item in items:
            self._id_to_idx[item.item_id] = len(self.items)
            self.items.append(item)
        self.embeddings = None

    def build(self) -> None:
        """构建向量索引"""
        if not self.items:
            self.logger.warning("No items to index")
            return

        embeddings = [item.embedding for item in self.items]
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # 归一化用于余弦相似度
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-10)

        self.logger.info(f"Built index with {len(self.items)} items")

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[IndexItem, float]]:
        """
        搜索最相似的项

        Args:
            query_embedding: 查询向量
            k: 返回数量
            filter_fn: 过滤函数

        Returns:
            (IndexItem, score) 列表
        """
        if self.embeddings is None:
            self.build()

        if self.embeddings is None or len(self.items) == 0:
            return []

        # 归一化查询向量
        query = np.array(query_embedding, dtype=np.float32)
        query = query / np.maximum(np.linalg.norm(query), 1e-10)

        # 计算相似度
        similarities = np.dot(self.embeddings, query)

        # 应用过滤器: 不匹配项设为 -2 (低于任何有效余弦相似度), 避免浪费 k 槽位
        if filter_fn:
            mask = np.array([filter_fn(item) for item in self.items])
            similarities = np.where(mask, similarities, -2.0)

        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] > -1.0:  # 排除被过滤项 (-2.0)
                results.append((self.items[idx], float(similarities[idx])))

        return results

    def save(self, path: str) -> None:
        """保存索引"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "items": [
                {
                    "item_id": item.item_id,
                    "content": item.content,
                    "embedding": item.embedding,
                    "metadata": item.metadata
                }
                for item in self.items
            ]
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Saved index to {path}")

    def load(self, path: str) -> None:
        """加载索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.dimension = data["dimension"]
        self.items = [
            IndexItem(
                item_id=item["item_id"],
                content=item["content"],
                embedding=item["embedding"],
                metadata=item["metadata"]
            )
            for item in data["items"]
        ]

        self._id_to_idx = {item.item_id: i for i, item in enumerate(self.items)}
        self.build()

        self.logger.info(f"Loaded index from {path} with {len(self.items)} items")


class HierarchicalIndexer(LoggerMixin):
    """
    层次化索引构建器
    构建三级索引结构：文档 -> 章节 -> 元素
    """

    def __init__(
        self,
        config: Optional[HMRConfig] = None,
        index_dir: Optional[str] = None
    ):
        self.config = config or get_config().hmr
        self.index_dir = Path(index_dir or "./data/embeddings")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 获取嵌入器
        self._text_embedder = None
        self._image_embedder = None

        # 三级索引
        self.doc_index: Optional[VectorIndex] = None
        self.section_index: Optional[VectorIndex] = None
        self.text_index: Optional[VectorIndex] = None
        self.image_index: Optional[VectorIndex] = None

        # 映射关系
        self._doc_to_sections: Dict[str, List[str]] = {}
        self._section_to_elements: Dict[str, List[str]] = {}

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            self._text_embedder = Embedder.get_text_embedder()
        return self._text_embedder

    @property
    def image_embedder(self):
        if self._image_embedder is None:
            self._image_embedder = Embedder.get_image_embedder()
        return self._image_embedder

    def build_index(
        self,
        documents: List[Document],
        sections: List[Section],
        chunks: List[Chunk],
        images: List[ImageElement]
    ) -> None:
        """
        构建完整的层次化索引

        Args:
            documents: 文档列表
            sections: 章节列表
            chunks: 文本块列表
            images: 图像列表
        """
        self.logger.info("Building hierarchical index...")

        # Level 1: 文档级索引
        self._build_document_index(documents)

        # Level 2: 章节级索引
        self._build_section_index(sections)

        # Level 3: 元素级索引
        self._build_element_index(chunks, images)

        # 构建映射关系
        self._build_mappings(documents, sections, chunks, images)

        self.logger.info("Hierarchical index built successfully")

    def _build_document_index(self, documents: List[Document]) -> None:
        """构建文档级索引"""
        self.logger.info(f"Building document index for {len(documents)} documents")

        dimension = self.text_embedder.dimension
        self.doc_index = VectorIndex(dimension)

        for doc in documents:
            # 使用文档摘要作为嵌入内容
            content = doc.summary or doc.name
            embedding = self.text_embedder.embed_text(content)

            item = IndexItem(
                item_id=doc.doc_id,
                content=content,
                embedding=embedding,
                metadata={
                    "name": doc.name,
                    "path": doc.path,
                    "num_pages": doc.num_pages
                }
            )
            self.doc_index.add(item)

        self.doc_index.build()
        self.doc_index.save(str(self.index_dir / "doc_index.pkl"))

    def _build_section_index(self, sections: List[Section]) -> None:
        """构建章节级索引"""
        self.logger.info(f"Building section index for {len(sections)} sections")

        dimension = self.text_embedder.dimension
        self.section_index = VectorIndex(dimension)

        for section in sections:
            # 使用标题+摘要作为嵌入内容
            content = f"{section.title}\n{section.abstract or ''}"
            embedding = self.text_embedder.embed_text(content)

            item = IndexItem(
                item_id=section.section_id,
                content=content,
                embedding=embedding,
                metadata={
                    "doc_id": section.doc_id,
                    "title": section.title,
                    "level": section.level,
                    "start_page": section.start_page,
                    "end_page": section.end_page
                }
            )
            self.section_index.add(item)

        self.section_index.build()
        self.section_index.save(str(self.index_dir / "section_index.pkl"))

    def _build_element_index(
        self,
        chunks: List[Chunk],
        images: List[ImageElement]
    ) -> None:
        """构建元素级索引（文本+图像）"""
        # 文本索引
        self.logger.info(f"Building text index for {len(chunks)} chunks")
        dimension = self.text_embedder.dimension
        self.text_index = VectorIndex(dimension)

        # 批量嵌入文本
        texts = [chunk.content for chunk in chunks]
        embeddings = self.text_embedder.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            item = IndexItem(
                item_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata={
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "page": chunk.page,
                    "modality": "text",
                    **chunk.metadata
                }
            )
            self.text_index.add(item)

        self.text_index.build()
        self.text_index.save(str(self.index_dir / "text_index.pkl"))

        # 图像索引
        if images:
            self.logger.info(f"Building image index for {len(images)} images")
            image_dimension = self.image_embedder.dimension
            self.image_index = VectorIndex(image_dimension)

            for image in images:
                # 使用图像摘要嵌入或直接图像嵌入
                if image.summary:
                    embedding = self.text_embedder.embed_text(image.summary)
                    # 需要调整维度匹配
                    if len(embedding) != image_dimension:
                        # 使用CLIP嵌入
                        try:
                            embedding = self.image_embedder.embed_image(image.image_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to embed image: {e}")
                            continue
                else:
                    try:
                        embedding = self.image_embedder.embed_image(image.image_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to embed image: {e}")
                        continue

                item = IndexItem(
                    item_id=image.image_id,
                    content=image.summary or image.caption or "",
                    embedding=embedding,
                    metadata={
                        "doc_id": image.doc_id,
                        "section_id": image.section_id,
                        "page": image.page,
                        "image_path": image.image_path,
                        "image_type": image.image_type,
                        "modality": "image"
                    }
                )
                self.image_index.add(item)

            self.image_index.build()
            self.image_index.save(str(self.index_dir / "image_index.pkl"))

    def _build_text_only_index(self, chunks: List[Chunk]) -> None:
        """仅构建文本索引（跳过图像）"""
        self.logger.info(f"Building text-only index for {len(chunks)} chunks")
        dimension = self.text_embedder.dimension
        self.text_index = VectorIndex(dimension)

        # 批量嵌入文本
        texts = [chunk.content for chunk in chunks]
        embeddings = self.text_embedder.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            item = IndexItem(
                item_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata={
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "page": chunk.page,
                    "modality": "text",
                    **chunk.metadata
                }
            )
            self.text_index.add(item)

        self.text_index.build()
        self.text_index.save(str(self.index_dir / "text_index.pkl"))
        self.logger.info(f"Text index saved to {self.index_dir / 'text_index.pkl'}")

    def _build_mappings(
        self,
        documents: List[Document],
        sections: List[Section],
        chunks: List[Chunk],
        images: List[ImageElement]
    ) -> None:
        """构建层次映射关系"""
        # 文档 -> 章节
        for section in sections:
            if section.doc_id not in self._doc_to_sections:
                self._doc_to_sections[section.doc_id] = []
            self._doc_to_sections[section.doc_id].append(section.section_id)

        # 章节 -> 元素
        for chunk in chunks:
            section_id = chunk.section_id or chunk.doc_id
            if section_id not in self._section_to_elements:
                self._section_to_elements[section_id] = []
            self._section_to_elements[section_id].append(chunk.chunk_id)

        for image in images:
            section_id = image.section_id or image.doc_id
            if section_id not in self._section_to_elements:
                self._section_to_elements[section_id] = []
            self._section_to_elements[section_id].append(image.image_id)

        # 保存映射
        mappings = {
            "doc_to_sections": self._doc_to_sections,
            "section_to_elements": self._section_to_elements
        }
        with open(self.index_dir / "mappings.json", 'w') as f:
            json.dump(mappings, f)

    def load_index(self) -> None:
        """加载已有索引"""
        self.logger.info("Loading hierarchical index...")

        # 加载各级索引
        doc_path = self.index_dir / "doc_index.pkl"
        if doc_path.exists():
            self.doc_index = VectorIndex(0)
            self.doc_index.load(str(doc_path))

        section_path = self.index_dir / "section_index.pkl"
        if section_path.exists():
            self.section_index = VectorIndex(0)
            self.section_index.load(str(section_path))

        text_path = self.index_dir / "text_index.pkl"
        if text_path.exists():
            self.text_index = VectorIndex(0)
            self.text_index.load(str(text_path))

        image_path = self.index_dir / "image_index.pkl"
        if image_path.exists():
            self.image_index = VectorIndex(0)
            self.image_index.load(str(image_path))

        # 加载映射
        mappings_path = self.index_dir / "mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self._doc_to_sections = mappings.get("doc_to_sections", {})
                self._section_to_elements = mappings.get("section_to_elements", {})

        self.logger.info("Hierarchical index loaded")

    def get_sections_for_docs(self, doc_ids: List[str]) -> List[str]:
        """获取文档下的章节ID"""
        section_ids = []
        for doc_id in doc_ids:
            section_ids.extend(self._doc_to_sections.get(doc_id, []))
        return section_ids

    def get_elements_for_sections(self, section_ids: List[str]) -> List[str]:
        """获取章节下的元素ID"""
        element_ids = []
        for section_id in section_ids:
            element_ids.extend(self._section_to_elements.get(section_id, []))
        return element_ids
