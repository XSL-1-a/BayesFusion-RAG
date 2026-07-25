"""
HEA-MRAG 数据类型定义
定义项目中使用的所有核心数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# 枚举类型
# =============================================================================

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"


class EntityType(Enum):
    """实体类型"""
    COMPONENT = "Component"
    PARAMETER = "Parameter"
    PROCEDURE = "Procedure"
    CONSTRAINT = "Constraint"
    STANDARD = "Standard"
    INTERFACE = "Interface"


class ModalityPreferenceType(Enum):
    """模态偏好类型"""
    TEXT_DOMINANT = "TEXT_DOMINANT"
    IMAGE_DOMINANT = "IMAGE_DOMINANT"
    BALANCED = "BALANCED"


class RetrievalLevel(Enum):
    """检索层级"""
    DOCUMENT = "document"
    SECTION = "section"
    ELEMENT = "element"


# =============================================================================
# 文档相关数据类型
# =============================================================================

@dataclass
class SourceMetadata:
    """来源元数据"""
    document: str
    page: int
    section: Optional[str] = None
    paragraph: Optional[int] = None
    bounding_box: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "paragraph": self.paragraph,
            "bounding_box": self.bounding_box
        }


@dataclass
class Document:
    """文档"""
    doc_id: str
    name: str
    path: str
    num_pages: int
    summary: Optional[str] = None
    sections: List["Section"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "path": self.path,
            "num_pages": self.num_pages,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata
        }


@dataclass
class Section:
    """章节"""
    section_id: str
    doc_id: str
    title: str
    level: int  # 1, 2, 3... 表示标题级别
    start_page: int
    end_page: int
    abstract: Optional[str] = None
    content: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "abstract": self.abstract,
            "content": self.content
        }


@dataclass
class Chunk:
    """文本块"""
    chunk_id: str
    doc_id: str
    section_id: Optional[str]
    content: str
    modality: ModalityType
    page: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "content": self.content,
            "modality": self.modality.value,
            "page": self.page,
            "metadata": self.metadata
        }


@dataclass
class ImageElement:
    """图像元素"""
    image_id: str
    doc_id: str
    section_id: Optional[str]
    page: int
    image_path: str
    image_type: str  # diagram, photo, chart, table_image
    caption: Optional[str] = None
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    bounding_box: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "page": self.page,
            "image_path": self.image_path,
            "image_type": self.image_type,
            "caption": self.caption,
            "summary": self.summary,
            "metadata": self.metadata
        }


# =============================================================================
# 实体相关数据类型
# =============================================================================

@dataclass
class Entity:
    """实体"""
    entity_id: str
    entity_type: EntityType
    entity_name: str
    attributes: Dict[str, Any]
    source: SourceMetadata
    evidence_text: str
    confidence: float
    extraction_modality: ModalityType = ModalityType.TEXT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "entity_name": self.entity_name,
            "attributes": self.attributes,
            "source": self.source.to_dict(),
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "extraction_modality": self.extraction_modality.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            entity_id=data["entity_id"],
            entity_type=EntityType(data["entity_type"]),
            entity_name=data["entity_name"],
            attributes=data["attributes"],
            source=SourceMetadata(**data["source"]),
            evidence_text=data["evidence_text"],
            confidence=data["confidence"],
            extraction_modality=ModalityType(data.get("extraction_modality", "text"))
        )


@dataclass
class Relation:
    """实体关系"""
    relation_id: str
    source_entity: str  # entity_name
    relation_type: str
    target_entity: str  # entity_name
    confidence: float
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_entity": self.source_entity,
            "relation_type": self.relation_type,
            "target_entity": self.target_entity,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


# =============================================================================
# 检索相关数据类型
# =============================================================================

@dataclass
class RetrievalResult:
    """检索结果"""
    item_id: str
    content: str
    modality: ModalityType
    score: float
    source: SourceMetadata
    level: RetrievalLevel
    retrieval_path: Optional[List[str]] = None  # [doc_id, section_id, chunk_id]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "modality": self.modality.value,
            "score": self.score,
            "source": self.source.to_dict(),
            "level": self.level.value,
            "retrieval_path": self.retrieval_path
        }


@dataclass
class ModalityPreference:
    """模态偏好"""
    preference_type: ModalityPreferenceType
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_type": self.preference_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


# =============================================================================
# 生成相关数据类型
# =============================================================================

@dataclass
class Citation:
    """引用"""
    citation_id: int
    claim: str
    source: SourceMetadata
    position: int  # 在答案文本中的位置

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "claim": self.claim,
            "source": self.source.to_dict(),
            "position": self.position
        }


@dataclass
class VerifiedCitation(Citation):
    """已验证的引用"""
    verified: bool = False
    evidence: Optional[str] = None
    verification_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "verified": self.verified,
            "evidence": self.evidence,
            "verification_reason": self.verification_reason
        })
        return base


@dataclass
class TraceableAnswer:
    """可追溯答案"""
    query: str
    answer: str
    citations: List[VerifiedCitation]
    sources: List[RetrievalResult]
    confidence: float
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "generation_time": self.generation_time,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# 评估相关数据类型
# =============================================================================

@dataclass
class EvaluationResult:
    """评估结果"""
    query_id: str
    query: str
    method: str
    metrics: Dict[str, float]
    answer: TraceableAnswer
    ground_truth: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "method": self.method,
            "metrics": self.metrics,
            "answer": self.answer.to_dict(),
            "ground_truth": self.ground_truth,
            "timestamp": self.timestamp
        }


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    description: str
    methods: List[str]
    dataset: str
    metrics: List[str]
    num_trials: int = 3
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "methods": self.methods,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "num_trials": self.num_trials,
            "seed": self.seed
        }
