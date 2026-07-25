"""
工业实体抽取器
基于客户 实体感知.md 的定义，实现规则+LLM混合抽取

抽取流程:
1. 规则抽取: 使用正则表达式快速识别明确的实体
2. LLM抽取: 使用LLM识别复杂的、上下文相关的实体
3. 实体融合: 合并去重，保留最高置信度的结果
4. 关系抽取: 识别实体间的关系
"""

import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from .industrial_ontology import (
    IndustrialEntityOntology,
    EntityType,
    RelationType,
    INDUSTRIAL_ENTITY_ONTOLOGY,
    RELATION_DEFINITIONS
)


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    entity_id: str
    entity_type: str
    entity_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence_text: str = ""
    confidence: float = 0.5
    extraction_method: str = "rule"  # rule, llm, hybrid
    source_document: str = ""
    source_page: int = 0
    source_section: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    relation_id: str
    source_entity: str
    relation_type: str
    target_entity: str
    confidence: float = 0.5
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndustrialEntityExtractor:
    """
    工业实体抽取器

    支持三种抽取模式:
    - rule_only: 仅使用规则抽取(快速，精确度高，召回率低)
    - llm_only: 仅使用LLM抽取(慢，召回率高，可能有幻觉)
    - hybrid: 混合模式(推荐，平衡精确度和召回率)
    """

    # LLM实体抽取Prompt模板
    ENTITY_EXTRACTION_PROMPT = """You are an expert in industrial document analysis.
Extract structured entities from the following technical content.

{ontology_description}

=== Content to Analyze ===
{content}

=== Source Information ===
Document: {document}
Page: {page}
Section: {section}

=== Instructions ===
1. Extract entities that are EXPLICITLY mentioned in the content
2. For each entity, provide the EXACT quote from the text as evidence
3. Assign confidence scores based on clarity:
   - 0.9-1.0: Explicitly named with clear type
   - 0.7-0.9: Clearly mentioned but type inferred
   - 0.5-0.7: Implied or partially mentioned
4. Do NOT infer entities that are not explicitly stated
5. Focus on entities relevant to maintenance/industrial context

=== Output Format (JSON) ===
{{
    "entities": [
        {{
            "entity_type": "Component|Parameter|Fault|Procedure|Resource|Constraint|Standard|Interface",
            "entity_name": "canonical name",
            "attributes": {{}},
            "evidence_text": "exact quote from content",
            "confidence": 0.0-1.0
        }}
    ]
}}

Extract entities now:"""

    # 关系抽取Prompt模板
    RELATION_EXTRACTION_PROMPT = """Given the following entities extracted from a technical document,
identify relationships between them.

=== Extracted Entities ===
{entities}

=== Original Text ===
{text}

=== Relation Types ===
{relation_types}

=== Instructions ===
1. Only identify relations with CLEAR evidence in the text
2. Provide exact quote as evidence for each relation
3. Assign confidence based on clarity of the relationship

=== Output Format (JSON) ===
{{
    "relations": [
        {{
            "source_entity": "entity name",
            "relation_type": "has_parameter|constrained_by|follows_standard|connects_to|part_of|requires_procedure|defined_in|caused_by|requires_resource|related_to",
            "target_entity": "entity name",
            "confidence": 0.0-1.0,
            "evidence": "exact quote"
        }}
    ]
}}

Extract relations now:"""

    def __init__(
        self,
        mode: str = "hybrid",
        confidence_threshold: float = 0.5,
        llm_client=None
    ):
        """
        初始化抽取器

        Args:
            mode: 抽取模式 (rule_only, llm_only, hybrid)
            confidence_threshold: 置信度阈值
            llm_client: LLM客户端 (可选，用于LLM抽取)
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.llm_client = llm_client
        self.ontology = IndustrialEntityOntology()

        # 预编译参数抽取正则
        self._param_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(V|A|W|Hz|kHz|MHz|GHz|Ω|ohm|psi|bar|Pa|kPa|MPa|°C|°F|K|mm|cm|m|km|in|ft|kg|g|lb|oz|s|ms|μs|ns|min|hr|%)',
            re.IGNORECASE
        )

        # 标准编号正则 - 改进：确保匹配完整编号(包括后缀如 -4A)
        self._standard_pattern = re.compile(
            r'\b(ISO|MIL-STD|MIL-SPEC|ASME|SAE|IEC|IEEE|RTCA|ANSI|ASTM|AN|MS|NAS)[\s-]?[\w\d][\w\d.-]*\b',
            re.IGNORECASE
        )

        # 过于笼统的实体名黑名单 - 这些词单独不构成有意义的实体
        self._generic_name_blacklist = {
            # Component 类
            'unit', 'component', 'part', 'device', 'assembly', 'element',
            'equipment', 'system', 'item', 'piece', 'section',
            # Procedure 类
            'procedure', 'process', 'method', 'step', 'instructions',
            'operation', 'task',
            # Resource 类
            'tool', 'replacement', 'supply', 'material',
            # 其他
            'type', 'kind', 'model', 'series',
        }

        # 参数上下文排除模式 - 排除任务编号、列表编号等
        self._param_exclude_context = re.compile(
            r'(?:task|step|item|figure|fig|table|para(?:graph)?|chapter|section|page)\s*'
            r'(?:number|no\.?|#)?\s*$',
            re.IGNORECASE
        )

    def extract(
        self,
        text: str,
        document: str = "",
        page: int = 0,
        section: str = ""
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        从文本中抽取实体和关系

        Args:
            text: 待抽取的文本
            document: 文档名称
            page: 页码
            section: 章节

        Returns:
            (实体列表, 关系列表)
        """
        all_entities = []
        all_relations = []

        # 1. 规则抽取
        if self.mode in ["rule_only", "hybrid"]:
            rule_entities = self._extract_by_rules(text, document, page, section)
            all_entities.extend(rule_entities)

        # 2. LLM抽取
        if self.mode in ["llm_only", "hybrid"] and self.llm_client:
            llm_entities = self._extract_by_llm(text, document, page, section)
            all_entities.extend(llm_entities)

        # 3. 实体融合去重
        merged_entities = self._merge_entities(all_entities)

        # 4. 过滤低置信度
        filtered_entities = [
            e for e in merged_entities
            if e.confidence >= self.confidence_threshold
        ]

        # 5. 关系抽取
        if len(filtered_entities) >= 2:
            all_relations = self._extract_relations(filtered_entities, text)

        return filtered_entities, all_relations

    def _extract_by_rules(
        self,
        text: str,
        document: str,
        page: int,
        section: str
    ) -> List[ExtractedEntity]:
        """
        基于规则的实体抽取
        """
        entities = []
        seen_names: Set[str] = set()

        # 使用本体的模式匹配
        matches = self.ontology.match_entity_type(text)

        for entity_type, match_text, start, end in matches:
            # 规范化实体名称
            entity_name = self._normalize_entity_name(match_text)

            # 质量过滤：跳过过于笼统的实体名
            if not self._is_valid_entity_name(entity_name, entity_type):
                continue

            # 获取上下文作为证据
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            evidence = text[context_start:context_end].strip()

            # 去重
            name_key = f"{entity_type}:{entity_name.lower()}"
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

            entity = ExtractedEntity(
                entity_id=self._generate_id(entity_type, entity_name),
                entity_type=entity_type,
                entity_name=entity_name,
                attributes={},
                evidence_text=evidence,
                confidence=0.85,  # 规则匹配的置信度
                extraction_method="rule",
                source_document=document,
                source_page=page,
                source_section=section
            )
            entities.append(entity)

        # 专门抽取参数值
        param_entities = self._extract_parameters(text, document, page, section)
        entities.extend(param_entities)

        # 专门抽取标准编号
        standard_entities = self._extract_standards(text, document, page, section)
        entities.extend(standard_entities)

        return entities

    def _extract_parameters(
        self,
        text: str,
        document: str,
        page: int,
        section: str
    ) -> List[ExtractedEntity]:
        """抽取参数值"""
        entities = []
        seen = set()

        for match in self._param_pattern.finditer(text):
            value = match.group(1)
            unit = match.group(2)

            # 获取参数名称的上下文
            start = max(0, match.start() - 50)
            context = text[start:match.start()].strip()

            # 排除任务编号、步骤编号等误匹配
            if self._is_reference_number(context, value, text, match.start()):
                continue

            # 尝试从上下文中提取参数名称
            param_name = self._extract_param_name(context, value, unit)

            # 排除名称为 "Parameter (X unit)" 的默认回退 - 说明没找到有意义的参数名
            if param_name.startswith("Parameter ("):
                continue

            key = f"{param_name}:{value}{unit}"
            if key in seen:
                continue
            seen.add(key)

            entity = ExtractedEntity(
                entity_id=self._generate_id("Parameter", key),
                entity_type="Parameter",
                entity_name=param_name,
                attributes={
                    "value": value,
                    "unit": unit,
                    "raw_text": match.group()
                },
                evidence_text=text[max(0, match.start()-30):min(len(text), match.end()+30)],
                confidence=0.9,
                extraction_method="rule",
                source_document=document,
                source_page=page,
                source_section=section
            )
            entities.append(entity)

        return entities

    def _extract_standards(
        self,
        text: str,
        document: str,
        page: int,
        section: str
    ) -> List[ExtractedEntity]:
        """抽取标准编号"""
        entities = []
        seen = set()

        for match in self._standard_pattern.finditer(text):
            standard_name = match.group().strip()

            if standard_name in seen:
                continue
            seen.add(standard_name)

            # 获取上下文
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)

            entity = ExtractedEntity(
                entity_id=self._generate_id("Standard", standard_name),
                entity_type="Standard",
                entity_name=standard_name,
                attributes={
                    "standard_org": match.group(1) if match.lastindex >= 1 else ""
                },
                evidence_text=text[start:end],
                confidence=0.95,  # 标准编号匹配置信度很高
                extraction_method="rule",
                source_document=document,
                source_page=page,
                source_section=section
            )
            entities.append(entity)

        return entities

    def _extract_by_llm(
        self,
        text: str,
        document: str,
        page: int,
        section: str
    ) -> List[ExtractedEntity]:
        """
        基于LLM的实体抽取
        """
        if not self.llm_client:
            return []

        # 构建Prompt
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(
            ontology_description=self.ontology.get_extraction_prompt(),
            content=text[:4000],  # 限制长度
            document=document or "Unknown",
            page=page,
            section=section or "N/A"
        )

        try:
            # 调用LLM
            response = self.llm_client.generate(prompt)

            # 解析JSON响应
            entities_data = self._parse_json_response(response)

            entities = []
            for raw in entities_data.get("entities", []):
                try:
                    entity = ExtractedEntity(
                        entity_id=self._generate_id(
                            raw.get("entity_type", "Unknown"),
                            raw.get("entity_name", "")
                        ),
                        entity_type=raw.get("entity_type", "Unknown"),
                        entity_name=raw.get("entity_name", ""),
                        attributes=raw.get("attributes", {}),
                        evidence_text=raw.get("evidence_text", ""),
                        confidence=raw.get("confidence", 0.5),
                        extraction_method="llm",
                        source_document=document,
                        source_page=page,
                        source_section=section
                    )
                    entities.append(entity)
                except Exception as e:
                    print(f"Warning: Failed to parse entity: {e}")
                    continue

            return entities

        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return []

    def _extract_relations(
        self,
        entities: List[ExtractedEntity],
        text: str
    ) -> List[ExtractedRelation]:
        """
        抽取实体间的关系

        使用规则 + 启发式方法
        """
        relations = []

        # 1. 基于规则的关系抽取
        rule_relations = self._extract_relations_by_rules(entities, text)
        relations.extend(rule_relations)

        # 2. 基于共现的关系推断
        cooc_relations = self._extract_relations_by_cooccurrence(entities, text)
        relations.extend(cooc_relations)

        # 去重
        seen = set()
        unique_relations = []
        for rel in relations:
            key = f"{rel.source_entity}:{rel.relation_type}:{rel.target_entity}"
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def _extract_relations_by_rules(
        self,
        entities: List[ExtractedEntity],
        text: str
    ) -> List[ExtractedRelation]:
        """基于规则的关系抽取"""
        relations = []

        # 构建实体名称到实体的映射
        entity_map = {e.entity_name.lower(): e for e in entities}

        # 关系模式
        relation_patterns = [
            # has_parameter: "X has a Y of Z" 或 "X rated at Y"
            (r'(\w+)\s+(?:has|with|rated\s+at)\s+.*?(\d+\s*\w+)', "has_parameter"),
            # part_of: "X is part of Y" 或 "X in Y"
            (r'(\w+)\s+(?:is\s+)?(?:part\s+of|in|within|inside)\s+(\w+)', "part_of"),
            # requires_procedure: "X requires Y"
            (r'(\w+)\s+(?:requires?|needs?)\s+(\w+)', "requires_procedure"),
            # connects_to: "X connects to Y"
            (r'(\w+)\s+(?:connects?\s+to|interfaces?\s+with|attached\s+to)\s+(\w+)', "connects_to"),
            # follows_standard: "per X" 或 "according to X"
            (r'(?:per|according\s+to|compliant\s+with)\s+([\w-]+)', "follows_standard"),
        ]

        for pattern, rel_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0].lower()
                    target = groups[1].lower() if len(groups) > 1 else ""

                    # 检查是否匹配已知实体
                    source_entity = self._find_matching_entity(source, entity_map)
                    target_entity = self._find_matching_entity(target, entity_map)

                    if source_entity and target_entity:
                        relation = ExtractedRelation(
                            relation_id=self._generate_id(
                                f"{source_entity}_{rel_type}_{target_entity}", ""
                            ),
                            source_entity=source_entity,
                            relation_type=rel_type,
                            target_entity=target_entity,
                            confidence=0.7,
                            evidence=match.group()
                        )
                        relations.append(relation)

        return relations

    def _extract_relations_by_cooccurrence(
        self,
        entities: List[ExtractedEntity],
        text: str,
        window_size: int = 100
    ) -> List[ExtractedRelation]:
        """基于共现的关系推断"""
        relations = []

        # 按位置排序实体
        entity_positions = []
        for entity in entities:
            pos = text.lower().find(entity.entity_name.lower())
            if pos >= 0:
                entity_positions.append((pos, entity))

        entity_positions.sort(key=lambda x: x[0])

        # 检查窗口内的共现
        for i, (pos1, e1) in enumerate(entity_positions):
            for pos2, e2 in entity_positions[i+1:]:
                if pos2 - pos1 > window_size:
                    break

                # 推断关系类型
                rel_type = self._infer_relation_type(e1.entity_type, e2.entity_type)
                if rel_type:
                    # 获取两个实体之间的文本作为证据
                    evidence = text[pos1:pos2 + len(e2.entity_name)][:200]

                    relation = ExtractedRelation(
                        relation_id=self._generate_id(
                            f"{e1.entity_name}_{rel_type}_{e2.entity_name}", ""
                        ),
                        source_entity=e1.entity_name,
                        relation_type=rel_type,
                        target_entity=e2.entity_name,
                        confidence=0.5,  # 共现推断置信度较低
                        evidence=evidence
                    )
                    relations.append(relation)

        return relations

    def _infer_relation_type(self, source_type: str, target_type: str) -> Optional[str]:
        """根据实体类型推断可能的关系类型"""
        # 常见的类型到关系映射
        type_relation_map = {
            ("Component", "Parameter"): "has_parameter",
            ("Component", "Constraint"): "constrained_by",
            ("Component", "Standard"): "follows_standard",
            ("Component", "Interface"): "connects_to",
            ("Component", "Component"): "part_of",
            ("Component", "Procedure"): "requires_procedure",
            ("Procedure", "Resource"): "requires_resource",
            ("Fault", "Component"): "caused_by",
            ("Fault", "Procedure"): "requires_procedure",
        }

        return type_relation_map.get((source_type, target_type))

    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """合并去重实体"""
        if not entities:
            return []

        merged = {}

        for entity in entities:
            key = f"{entity.entity_type}:{entity.entity_name.lower().strip()}"

            if key not in merged:
                merged[key] = entity
            else:
                existing = merged[key]
                # 保留置信度更高的
                if entity.confidence > existing.confidence:
                    # 合并属性
                    entity.attributes.update(existing.attributes)
                    merged[key] = entity
                else:
                    existing.attributes.update(entity.attributes)
                    # 如果方法不同，标记为混合
                    if entity.extraction_method != existing.extraction_method:
                        existing.extraction_method = "hybrid"

        return list(merged.values())

    def _is_valid_entity_name(self, name: str, entity_type: str) -> bool:
        """
        验证实体名称是否有效（不过于笼统）

        Returns:
            True if valid, False if should be filtered out
        """
        name_lower = name.lower().strip()

        # 长度检查
        if len(name_lower) < 2:
            return False

        # 黑名单检查 - 单独的笼统词
        if name_lower in self._generic_name_blacklist:
            return False

        # 纯数字不是有效实体名
        if name_lower.replace(' ', '').isdigit():
            return False

        # Constraint 类型：必须是短语（至少3个词），不能是单词
        if entity_type == "Constraint":
            word_count = len(name_lower.split())
            if word_count < 3:
                return False

        return True

    def _is_reference_number(self, context: str, value: str, full_text: str, match_pos: int) -> bool:
        """
        判断数字是否是引用编号（任务号、步骤号等）而非参数值

        Args:
            context: 数字前的上下文
            value: 数字值
            full_text: 完整文本
            match_pos: 匹配位置

        Returns:
            True if this is a reference number (should be excluded)
        """
        context_lower = context.lower().rstrip()

        # 检查上下文是否包含编号指示词
        ref_patterns = [
            r'task\s*$', r'step\s*$', r'item\s*$',
            r'figure\s*$', r'fig\.?\s*$', r'table\s*$',
            r'para(?:graph)?\.?\s*$', r'chapter\s*$',
            r'section\s*$', r'page\s*$', r'no\.?\s*$',
            r'#\s*$', r'number\s*$',
            r'perform\s+task\s*$', r'task\s+\d+[\s,]+(?:and\s+)?$',
        ]

        for pattern in ref_patterns:
            if re.search(pattern, context_lower):
                return True

        # 检查 "task X, Y and Z" 模式 (列表中的数字)
        # 向前看更多上下文
        broader_start = max(0, match_pos - 80)
        broader_context = full_text[broader_start:match_pos].lower()
        if re.search(r'task\s+\d+[\s,]+(?:and\s+)?(?:\d+[\s,]+(?:and\s+)?)*$', broader_context):
            return True

        # 检查 "X and Y as needed" 模式
        after_start = match_pos
        after_end = min(len(full_text), match_pos + 30)
        after_context = full_text[after_start:after_end].lower()
        if re.search(r'^\d+\s+(?:as\s+needed|if\s+needed|when\s+needed)', after_context):
            return True

        return False

    def _normalize_entity_name(self, name: str) -> str:
        """规范化实体名称"""
        # 去除多余空格
        name = " ".join(name.split())
        # 去除开头的冠词
        name = re.sub(r'^(?:the|a|an)\s+', '', name, flags=re.IGNORECASE)
        # 去除末尾的虚词
        name = re.sub(r'\s+(?:the|a|an|of|to|for|in|on|at|by)$', '', name, flags=re.IGNORECASE)
        # 首字母大写
        return name.title() if len(name) < 50 else name

    def _extract_param_name(self, context: str, value: str, unit: str) -> str:
        """从上下文中提取参数名称"""
        # 常见参数名称关键词
        param_keywords = [
            "voltage", "current", "power", "temperature", "pressure",
            "frequency", "resistance", "impedance", "capacitance",
            "weight", "mass", "length", "width", "height", "depth",
            "speed", "rpm", "torque", "flow", "rate"
        ]

        context_lower = context.lower()
        for keyword in param_keywords:
            if keyword in context_lower:
                return keyword.title()

        # 如果找不到，使用单位推断
        unit_to_param = {
            "V": "Voltage", "A": "Current", "W": "Power",
            "Hz": "Frequency", "Ω": "Resistance", "ohm": "Resistance",
            "°C": "Temperature", "°F": "Temperature", "K": "Temperature",
            "psi": "Pressure", "bar": "Pressure", "Pa": "Pressure",
            "kg": "Weight", "lb": "Weight", "g": "Weight",
            "mm": "Dimension", "cm": "Dimension", "m": "Dimension", "in": "Dimension"
        }

        return unit_to_param.get(unit, f"Parameter ({value} {unit})")

    def _find_matching_entity(
        self,
        name: str,
        entity_map: Dict[str, ExtractedEntity]
    ) -> Optional[str]:
        """在实体映射中查找匹配的实体"""
        name = name.lower().strip()

        # 精确匹配
        if name in entity_map:
            return entity_map[name].entity_name

        # 部分匹配
        for key, entity in entity_map.items():
            if name in key or key in name:
                return entity.entity_name

        return None

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的JSON响应"""
        # 尝试找到JSON部分
        start = response.find('{')
        end = response.rfind('}') + 1

        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        return {"entities": [], "relations": []}

    def _generate_id(self, type_or_prefix: str, name: str) -> str:
        """生成唯一ID"""
        content = f"{type_or_prefix}_{name}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# 便捷函数
def create_extractor(
    mode: str = "hybrid",
    confidence_threshold: float = 0.5,
    llm_client=None
) -> IndustrialEntityExtractor:
    """创建实体抽取器实例"""
    return IndustrialEntityExtractor(
        mode=mode,
        confidence_threshold=confidence_threshold,
        llm_client=llm_client
    )
