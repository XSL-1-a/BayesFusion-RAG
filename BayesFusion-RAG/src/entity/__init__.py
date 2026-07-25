"""
实体感知型结构化抽取模块 (EA-RAG - Entity-Aware RAG)
- 工业领域实体本体定义 (基于客户 实体感知.md)
- 多模态实体抽取 (规则+LLM混合)
- 轻量级JSON存储 / Neo4j图数据库
- 实体关系构建
"""

# 原有模块 (保持向后兼容)
from .ontology import IndustrialOntology
from .extractor import MultimodalEntityExtractor
from .store import LightweightEntityStore
from .graph_store import EntityKnowledgeGraph
from .entity_enhanced_rag import EntityEnhancedRAG

# 新模块 - 基于客户 实体感知.md 的实现
from .industrial_ontology import (
    IndustrialEntityOntology,
    EntityType,
    RelationType,
    EntityDefinition,
    RelationDefinition,
    INDUSTRIAL_ENTITY_ONTOLOGY,
    RELATION_DEFINITIONS,
    get_ontology
)
from .industrial_extractor import (
    IndustrialEntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    create_extractor
)

# 维修性专用本体
from .maintainability_ontology import (
    MAINTAINABILITY_ENTITY_ONTOLOGY,
    ENTITY_TYPES as MAINTAINABILITY_ENTITY_TYPES,
    RELATION_TYPES as MAINTAINABILITY_RELATION_TYPES,
    get_all_patterns as get_maintainability_patterns
)

__all__ = [
    # 原有模块
    "IndustrialOntology",
    "MultimodalEntityExtractor",
    "LightweightEntityStore",
    "EntityKnowledgeGraph",
    "EntityEnhancedRAG",
    # 新模块 - 基于客户定义
    "IndustrialEntityOntology",
    "IndustrialEntityExtractor",
    "EntityType",
    "RelationType",
    "EntityDefinition",
    "RelationDefinition",
    "ExtractedEntity",
    "ExtractedRelation",
    "INDUSTRIAL_ENTITY_ONTOLOGY",
    "RELATION_DEFINITIONS",
    "get_ontology",
    "create_extractor",
    # 维修性专用
    "MAINTAINABILITY_ENTITY_ONTOLOGY",
    "MAINTAINABILITY_ENTITY_TYPES",
    "MAINTAINABILITY_RELATION_TYPES",
    "get_maintainability_patterns",
]
