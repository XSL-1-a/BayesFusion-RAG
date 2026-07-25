"""
工业领域实体本体定义
定义工业文档中常见的实体类型和关系
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class EntityCategory(Enum):
    """实体类别"""
    COMPONENT = "Component"
    PARAMETER = "Parameter"
    PROCEDURE = "Procedure"
    CONSTRAINT = "Constraint"
    STANDARD = "Standard"
    INTERFACE = "Interface"


class RelationType(Enum):
    """关系类型"""
    HAS_PARAMETER = "has_parameter"
    CONSTRAINED_BY = "constrained_by"
    FOLLOWS_STANDARD = "follows_standard"
    CONNECTS_TO = "connects_to"
    PART_OF = "part_of"
    REQUIRES_PROCEDURE = "requires_procedure"
    DEFINED_IN = "defined_in"
    RELATED_TO = "related_to"


@dataclass
class EntityTypeDefinition:
    """实体类型定义"""
    name: str
    description: str
    examples: List[str]
    attributes: List[str]
    keywords: List[str] = field(default_factory=list)


class IndustrialOntology:
    """
    工业领域本体
    定义实体类型、关系类型和抽取规则
    """

    # 实体类型定义
    ENTITY_TYPES: Dict[str, EntityTypeDefinition] = {
        "Component": EntityTypeDefinition(
            name="Component",
            description="物理组件/部件，包括机械、电子或软件组件",
            examples=["valve", "sensor", "controller", "pump", "motor", "circuit board", "CPU"],
            attributes=["name", "type", "model", "manufacturer", "material", "dimensions"],
            keywords=["component", "part", "unit", "module", "assembly", "device", "element"]
        ),
        "Parameter": EntityTypeDefinition(
            name="Parameter",
            description="技术参数/规格，包括数值、范围和单位",
            examples=["temperature", "pressure", "dimension", "voltage", "current", "frequency"],
            attributes=["name", "value", "unit", "min", "max", "tolerance", "typical"],
            keywords=["parameter", "specification", "rating", "value", "range", "limit"]
        ),
        "Procedure": EntityTypeDefinition(
            name="Procedure",
            description="操作流程/维护步骤，包括安装、校准、故障排除等",
            examples=["installation", "calibration", "troubleshooting", "maintenance", "testing"],
            attributes=["name", "steps", "preconditions", "tools_required", "duration", "frequency"],
            keywords=["procedure", "process", "step", "instruction", "method", "operation"]
        ),
        "Constraint": EntityTypeDefinition(
            name="Constraint",
            description="设计约束/限制条件，包括操作范围、安全限制等",
            examples=["operating range", "safety limit", "environmental requirement", "clearance"],
            attributes=["name", "condition", "consequence", "severity", "applies_to"],
            keywords=["constraint", "limitation", "restriction", "requirement", "condition", "must"]
        ),
        "Standard": EntityTypeDefinition(
            name="Standard",
            description="引用的标准/规范，包括行业标准和法规",
            examples=["ISO 9001", "MIL-STD-810", "ASME Y14.5", "IEC 61131", "SAE J1939"],
            attributes=["name", "version", "section", "organization", "year"],
            keywords=["standard", "specification", "regulation", "code", "norm", "guideline"]
        ),
        "Interface": EntityTypeDefinition(
            name="Interface",
            description="接口/连接关系，包括电气、机械和数据接口",
            examples=["electrical interface", "mechanical coupling", "data port", "connector"],
            attributes=["name", "type", "protocol", "connected_components", "pin_count"],
            keywords=["interface", "connection", "port", "connector", "coupling", "link"]
        )
    }

    # 关系类型定义
    RELATION_TYPES: Dict[str, Dict[str, Any]] = {
        "has_parameter": {
            "description": "组件具有某个参数",
            "source_types": ["Component"],
            "target_types": ["Parameter"],
            "keywords": ["has", "with", "rated at", "specified as"]
        },
        "constrained_by": {
            "description": "组件受到某个约束",
            "source_types": ["Component", "Procedure"],
            "target_types": ["Constraint"],
            "keywords": ["limited by", "constrained by", "must not exceed", "should not"]
        },
        "follows_standard": {
            "description": "遵循某个标准",
            "source_types": ["Component", "Procedure", "Parameter"],
            "target_types": ["Standard"],
            "keywords": ["compliant with", "according to", "per", "as per", "in accordance with"]
        },
        "connects_to": {
            "description": "连接到另一个组件",
            "source_types": ["Component", "Interface"],
            "target_types": ["Component", "Interface"],
            "keywords": ["connects to", "linked to", "coupled with", "attached to"]
        },
        "part_of": {
            "description": "是某个组件的一部分",
            "source_types": ["Component"],
            "target_types": ["Component"],
            "keywords": ["part of", "component of", "included in", "belongs to"]
        },
        "requires_procedure": {
            "description": "需要某个操作流程",
            "source_types": ["Component"],
            "target_types": ["Procedure"],
            "keywords": ["requires", "needs", "must undergo", "should follow"]
        },
        "defined_in": {
            "description": "定义在某个来源中",
            "source_types": ["Component", "Parameter", "Procedure", "Constraint", "Standard", "Interface"],
            "target_types": ["Source"],
            "keywords": ["defined in", "specified in", "described in", "documented in"]
        }
    }

    @classmethod
    def get_entity_type(cls, type_name: str) -> EntityTypeDefinition:
        """获取实体类型定义"""
        return cls.ENTITY_TYPES.get(type_name)

    @classmethod
    def get_all_entity_types(cls) -> List[str]:
        """获取所有实体类型名称"""
        return list(cls.ENTITY_TYPES.keys())

    @classmethod
    def get_entity_schema(cls) -> Dict[str, Any]:
        """获取实体抽取的JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {
                                "type": "string",
                                "enum": list(cls.ENTITY_TYPES.keys())
                            },
                            "entity_name": {"type": "string"},
                            "attributes": {"type": "object"},
                            "evidence_text": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["entity_type", "entity_name", "evidence_text", "confidence"]
                    }
                }
            }
        }

    @classmethod
    def get_relation_schema(cls) -> Dict[str, Any]:
        """获取关系抽取的JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_entity": {"type": "string"},
                            "relation_type": {
                                "type": "string",
                                "enum": list(cls.RELATION_TYPES.keys())
                            },
                            "target_entity": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence": {"type": "string"}
                        },
                        "required": ["source_entity", "relation_type", "target_entity", "confidence"]
                    }
                }
            }
        }

    @classmethod
    def get_ontology_prompt(cls) -> str:
        """生成用于LLM的本体描述prompt"""
        lines = ["Entity Types and Definitions:\n"]

        for type_name, type_def in cls.ENTITY_TYPES.items():
            lines.append(f"\n{type_name}:")
            lines.append(f"  Description: {type_def.description}")
            lines.append(f"  Examples: {', '.join(type_def.examples[:5])}")
            lines.append(f"  Attributes: {', '.join(type_def.attributes)}")

        lines.append("\n\nRelation Types:")
        for rel_name, rel_def in cls.RELATION_TYPES.items():
            lines.append(f"\n{rel_name}:")
            lines.append(f"  Description: {rel_def['description']}")
            lines.append(f"  Source types: {', '.join(rel_def['source_types'])}")
            lines.append(f"  Target types: {', '.join(rel_def['target_types'])}")

        return "\n".join(lines)
