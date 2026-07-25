"""
工业领域实体本体定义
基于客户 实体感知.md 的定义

实体类型:
- Component: 物理组件/部件
- Parameter: 技术参数/规格
- Fault: 故障类型/故障现象
- Procedure: 操作流程/维护步骤
- Resource: 维修资源(备件、工具、耗材)
- Constraint: 设计约束/限制条件
- Standard: 标准/规范
- Interface: 接口/连接关系

关系类型:
- has_parameter: Component → Parameter
- constrained_by: Component → Constraint
- follows_standard: Component/Procedure → Standard
- connects_to: Component → Component (via Interface)
- part_of: Component → Component (hierarchy)
- requires_procedure: Component → Procedure
- defined_in: Entity → Source (document, section, page)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class EntityType(Enum):
    """实体类型枚举"""
    COMPONENT = "Component"
    PARAMETER = "Parameter"
    FAULT = "Fault"
    PROCEDURE = "Procedure"
    RESOURCE = "Resource"
    CONSTRAINT = "Constraint"
    STANDARD = "Standard"
    INTERFACE = "Interface"


class RelationType(Enum):
    """关系类型枚举"""
    HAS_PARAMETER = "has_parameter"
    CONSTRAINED_BY = "constrained_by"
    FOLLOWS_STANDARD = "follows_standard"
    CONNECTS_TO = "connects_to"
    PART_OF = "part_of"
    REQUIRES_PROCEDURE = "requires_procedure"
    DEFINED_IN = "defined_in"
    # 扩展关系
    CAUSED_BY = "caused_by"           # Fault → Component/Fault
    REQUIRES_RESOURCE = "requires_resource"  # Procedure → Resource
    RELATED_TO = "related_to"         # 通用关联


@dataclass
class EntityDefinition:
    """实体类型定义"""
    name: str
    description: str
    description_cn: str
    examples: List[str]
    attributes: List[str]
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class RelationDefinition:
    """关系类型定义"""
    name: str
    description: str
    source_types: List[str]
    target_types: List[str]
    keywords: List[str] = field(default_factory=list)


# ============================================================
# 工业实体本体定义 (基于客户 实体感知.md)
# ============================================================

INDUSTRIAL_ENTITY_ONTOLOGY: Dict[str, EntityDefinition] = {
    "Component": EntityDefinition(
        name="Component",
        description="Physical component/part",
        description_cn="物理组件/部件",
        examples=["valve", "sensor", "controller", "pump", "motor", "circuit board", "actuator"],
        attributes=["name", "type", "model", "manufacturer"],
        patterns=[
            # 具体组件名称 - 不匹配过于笼统的词
            r'\b(valve|sensor|controller|pump|motor|actuator|switch|relay)\b',
            r'\b(circuit\s*(?:board|card)|PCB|module)\b',
            r'\b(LRU|SRU|WRA)\b',
            # 带修饰语的组件 (如 "wheel assembly", "filter element")
            r'\b(\w+\s+(?:assembly|unit|component|device|element))\b',
        ],
        keywords=["component", "part", "unit", "module", "assembly", "device", "element", "equipment"]
    ),

    "Parameter": EntityDefinition(
        name="Parameter",
        description="Technical parameter/specification",
        description_cn="技术参数/规格",
        examples=["temperature", "pressure", "dimension", "voltage", "current", "frequency", "resistance"],
        attributes=["name", "value", "unit", "min", "max", "tolerance"],
        patterns=[
            r'(\d+(?:\.\d+)?)\s*(V|A|W|Hz|kHz|MHz|Ω|ohm|psi|bar|°C|°F|mm|cm|m|kg|lb|in)',
            r'\b(temperature|pressure|voltage|current|power|frequency|resistance|impedance)\b',
            r'\b(max(?:imum)?|min(?:imum)?|nominal|typical|rated)\s+\w+',
        ],
        keywords=["parameter", "specification", "rating", "value", "range", "limit", "tolerance"]
    ),

    "Fault": EntityDefinition(
        name="Fault",
        description="Fault type/phenomenon of equipment/component",
        description_cn="装备/部件的故障类型/故障现象",
        examples=["valve jamming", "sensor drift", "pressure overrun", "interface loose",
                  "short circuit", "open circuit", "leakage", "overheating"],
        attributes=["name", "phenomenon", "root_cause", "impact", "symptom_parameters"],
        patterns=[
            r'\b(fault|failure|malfunction|defect|anomaly)\b',
            r'\b(jamming|drift|overrun|stuck|broken|damaged|worn)\b',
            r'\b(short\s*circuit|open\s*circuit|leakage|overheating|corrosion)\b',
            # 移除 "does not" - 太模糊，误匹配率高
            r'\b(fails\s+to\s+\w+|unable\s+to\s+\w+|inoperative)\b',
        ],
        keywords=["fault", "failure", "malfunction", "defect", "error", "problem", "issue", "symptom"]
    ),

    "Procedure": EntityDefinition(
        name="Procedure",
        description="Operation procedure/maintenance steps",
        description_cn="操作流程/维护步骤",
        examples=["maintenance preparation", "fault diagnosis", "fault isolation",
                  "disassembly", "assembly", "calibration", "testing", "inspection"],
        attributes=["name", "steps", "preconditions", "tools_required"],
        patterns=[
            r'\b(procedure|process|steps|instructions|method)\b',
            r'\b(remove|install|replace|calibrate|align|adjust|inspect|test|verify)\b',
            r'\b(disassembly|assembly|maintenance|repair|service|overhaul)\b',
            r'\b(step\s*\d+|first|then|next|finally|after|before)\b',
        ],
        keywords=["procedure", "process", "step", "instruction", "method", "operation", "task"]
    ),

    "Resource": EntityDefinition(
        name="Resource",
        description="Maintenance resources: spare parts, tools, consumables",
        description_cn="维修所需的备件、工具、耗材",
        examples=["solenoid valve spare", "hydraulic wrench", "sealing gasket",
                  "lubricating oil", "multimeter", "torque wrench", "test set"],
        attributes=["name", "type", "model", "manufacturer", "applicable_components", "stock_status"],
        patterns=[
            r'\b(spare\s*(?:part)?|replacement|consumable)\b',
            r'\b(tool|wrench|screwdriver|pliers|meter|tester|gauge|kit)\b',
            r'\b(equipment|test\s*set|support\s*equipment|GSE|TMDE)\b',
            r'\b(lubricant|sealant|adhesive|gasket|O-ring|washer)\b',
        ],
        keywords=["tool", "equipment", "spare", "part", "consumable", "resource", "supply"]
    ),

    "Constraint": EntityDefinition(
        name="Constraint",
        description="Design constraint/limitation",
        description_cn="设计约束/限制条件",
        examples=["operating range", "safety limit", "environmental requirement",
                  "temperature limit", "weight limit", "clearance requirement"],
        attributes=["name", "condition", "consequence"],
        patterns=[
            # 只匹配完整的约束短语，不匹配单个关键词
            # 约束 = "限定词 + 具体条件"
            r'\b(must\s+(?:not\s+)?\w+\s+\w+)',               # "must not exceed ...", "must be ..."
            r'\b(shall\s+(?:not\s+)?\w+\s+\w+)',              # "shall not ..."
            r'\b(do\s+not\s+exceed\s+[\w\s]+)',               # "do not exceed ..."
            r'\b(not\s+(?:exceed|less\s+than|more\s+than)\s+[\w\s]+)',  # "not exceed 5000 psi"
            r'\b(maximum\s+(?:safe\s+)?\w+\s+(?:is|of)\s+[\w\s]+)',    # "maximum safe speed is 20 mph"
            r'\b(minimum\s+\w+\s+(?:is|of)\s+[\w\s]+)',       # "minimum clearance is ..."
        ],
        keywords=["constraint", "limitation", "restriction", "requirement", "condition", "must", "shall"]
    ),

    "Standard": EntityDefinition(
        name="Standard",
        description="Referenced standard/specification",
        description_cn="引用的标准/规范",
        examples=["ISO 9001", "MIL-STD-810", "ASME Y14.5", "SAE AS9100",
                  "IEC 61131", "IEEE 802", "RTCA DO-178"],
        attributes=["name", "version", "section"],
        patterns=[
            # 只匹配完整标准编号 (组织代号+数字编号)，不匹配单独关键词
            r'\b(ISO|MIL-STD|MIL-SPEC|ASME|SAE|IEC|IEEE|RTCA|ANSI|ASTM)[\s-]?\d[\w.-]*\b',
            r'\bAN\d[\w.-]+\b',   # AN6235-4A 格式
            r'\bMS\d[\w.-]+\b',   # MS 标准件
            r'\bNAS\d[\w.-]+\b',  # NAS 标准件
            # 移除 "standard", "specification" 等单独关键词 - 这些不是标准实体
            # 移除 "per", "according to" 等 - 这些是关系指示词
        ],
        keywords=["standard", "specification", "regulation", "code", "norm", "guideline", "compliance"]
    ),

    "Interface": EntityDefinition(
        name="Interface",
        description="Interface/connection relationship",
        description_cn="接口/连接关系",
        examples=["electrical interface", "mechanical coupling", "data port",
                  "connector", "terminal", "bus interface", "serial port"],
        attributes=["name", "type", "connected_components"],
        patterns=[
            # 具体的接口/连接器名词，不匹配动词短语
            r'\b(interface|connector|terminal|coupling|connection)\b',
            r'\b(plug|socket|pin|bus|cable|wire|harness)\b',
            r'\b(RS-232|RS-485|USB|Ethernet|CAN|ARINC|MIL-STD-1553)\b',
            # 移除 "port" (容易从 "report" 误匹配), "jack" (容易匹配动词)
            # 移除 "connects to/attached to" 等动词短语 - 这些是关系，不是实体
        ],
        keywords=["interface", "connection", "connector", "coupling", "link", "bus"]
    ),
}


# ============================================================
# 关系类型定义
# ============================================================

RELATION_DEFINITIONS: Dict[str, RelationDefinition] = {
    "has_parameter": RelationDefinition(
        name="has_parameter",
        description="Component has a parameter",
        source_types=["Component"],
        target_types=["Parameter"],
        keywords=["has", "with", "rated at", "specified as", "operates at"]
    ),

    "constrained_by": RelationDefinition(
        name="constrained_by",
        description="Component/Procedure is constrained by",
        source_types=["Component", "Procedure"],
        target_types=["Constraint"],
        keywords=["limited by", "constrained by", "must not exceed", "shall not"]
    ),

    "follows_standard": RelationDefinition(
        name="follows_standard",
        description="Follows a standard/specification",
        source_types=["Component", "Procedure"],
        target_types=["Standard"],
        keywords=["compliant with", "according to", "per", "in accordance with", "meets"]
    ),

    "connects_to": RelationDefinition(
        name="connects_to",
        description="Connects to another component via interface",
        source_types=["Component", "Interface"],
        target_types=["Component", "Interface"],
        keywords=["connects to", "linked to", "coupled with", "attached to", "interfaces with"]
    ),

    "part_of": RelationDefinition(
        name="part_of",
        description="Is part of a larger component (hierarchy)",
        source_types=["Component"],
        target_types=["Component"],
        keywords=["part of", "component of", "included in", "belongs to", "within"]
    ),

    "requires_procedure": RelationDefinition(
        name="requires_procedure",
        description="Component requires a procedure",
        source_types=["Component", "Fault"],
        target_types=["Procedure"],
        keywords=["requires", "needs", "must undergo", "should follow", "perform"]
    ),

    "defined_in": RelationDefinition(
        name="defined_in",
        description="Defined in a source document",
        source_types=["Component", "Parameter", "Procedure", "Constraint", "Standard", "Interface", "Fault", "Resource"],
        target_types=["Source"],
        keywords=["defined in", "specified in", "described in", "documented in", "see"]
    ),

    "caused_by": RelationDefinition(
        name="caused_by",
        description="Fault is caused by component or another fault",
        source_types=["Fault"],
        target_types=["Component", "Fault"],
        keywords=["caused by", "due to", "result of", "because of", "from"]
    ),

    "requires_resource": RelationDefinition(
        name="requires_resource",
        description="Procedure requires a resource (tool/spare/consumable)",
        source_types=["Procedure"],
        target_types=["Resource"],
        keywords=["requires", "needs", "using", "with", "tools required"]
    ),

    "related_to": RelationDefinition(
        name="related_to",
        description="General relationship between entities",
        source_types=["Component", "Parameter", "Procedure", "Constraint", "Standard", "Interface", "Fault", "Resource"],
        target_types=["Component", "Parameter", "Procedure", "Constraint", "Standard", "Interface", "Fault", "Resource"],
        keywords=["related to", "associated with", "relevant to", "see also"]
    ),
}


class IndustrialEntityOntology:
    """
    工业实体本体类
    提供实体类型定义、关系定义和抽取辅助方法
    """

    def __init__(self):
        self.entity_types = INDUSTRIAL_ENTITY_ONTOLOGY
        self.relation_types = RELATION_DEFINITIONS
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译所有正则表达式"""
        for entity_type, definition in self.entity_types.items():
            compiled = []
            for pattern in definition.patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    print(f"Warning: Invalid pattern in {entity_type}: {pattern} - {e}")
            self._compiled_patterns[entity_type] = compiled

    def get_entity_types(self) -> List[str]:
        """获取所有实体类型名称"""
        return list(self.entity_types.keys())

    def get_entity_definition(self, entity_type: str) -> Optional[EntityDefinition]:
        """获取实体类型定义"""
        return self.entity_types.get(entity_type)

    def get_relation_types(self) -> List[str]:
        """获取所有关系类型名称"""
        return list(self.relation_types.keys())

    def get_relation_definition(self, relation_type: str) -> Optional[RelationDefinition]:
        """获取关系类型定义"""
        return self.relation_types.get(relation_type)

    def match_entity_type(self, text: str) -> List[tuple]:
        """
        使用正则表达式匹配文本中的实体类型

        Args:
            text: 待匹配的文本

        Returns:
            List of (entity_type, match_text, start, end)
        """
        matches = []
        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append((
                        entity_type,
                        match.group(),
                        match.start(),
                        match.end()
                    ))
        return matches

    def get_extraction_prompt(self) -> str:
        """
        生成用于LLM实体抽取的本体描述Prompt
        """
        lines = [
            "=== Entity Types for Industrial Document Analysis ===\n"
        ]

        for etype, definition in self.entity_types.items():
            lines.append(f"\n**{etype}** ({definition.description_cn})")
            lines.append(f"  Description: {definition.description}")
            lines.append(f"  Examples: {', '.join(definition.examples[:5])}")
            lines.append(f"  Attributes: {', '.join(definition.attributes)}")

        lines.append("\n\n=== Relation Types ===\n")
        for rtype, definition in self.relation_types.items():
            lines.append(f"\n**{rtype}**")
            lines.append(f"  Description: {definition.description}")
            lines.append(f"  From: {', '.join(definition.source_types)}")
            lines.append(f"  To: {', '.join(definition.target_types)}")

        return "\n".join(lines)

    def get_entity_schema(self) -> Dict[str, Any]:
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
                                "enum": list(self.entity_types.keys())
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

    def get_relation_schema(self) -> Dict[str, Any]:
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
                                "enum": list(self.relation_types.keys())
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

    def validate_relation(self, source_type: str, relation_type: str, target_type: str) -> bool:
        """
        验证关系是否符合本体定义

        Args:
            source_type: 源实体类型
            relation_type: 关系类型
            target_type: 目标实体类型

        Returns:
            是否有效
        """
        rel_def = self.relation_types.get(relation_type)
        if not rel_def:
            return False

        return (source_type in rel_def.source_types and
                target_type in rel_def.target_types)


# 导出便捷函数
def get_ontology() -> IndustrialEntityOntology:
    """获取工业实体本体实例"""
    return IndustrialEntityOntology()
