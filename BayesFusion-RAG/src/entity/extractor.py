"""
多模态实体抽取模块
从文本和图像中抽取结构化实体
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..utils.config import get_config, EntityConfig
from ..utils.llm_client import LLMClient
from ..utils.data_types import (
    Entity, Relation, EntityType, ModalityType, SourceMetadata
)
from .ontology import IndustrialOntology


class MultimodalEntityExtractor(LoggerMixin):
    """
    多模态实体抽取器
    从文本和图像中抽取结构化实体和关系
    """

    EXTRACTION_PROMPT = """You are an expert in industrial document analysis.
Extract structured entities from the following technical content.

{ontology}

Content to analyze:
{content}

Source information:
- Document: {document}
- Page: {page}
- Section: {section}

Instructions:
1. Extract entities that are explicitly mentioned in the content
2. For each entity, provide the exact quote as evidence
3. Assign confidence scores based on clarity (0.5-1.0)
4. Do not infer entities that are not explicitly stated
5. Focus on technical entities relevant to industrial applications

Output a JSON object with the following structure:
{{
    "entities": [
        {{
            "entity_type": "Component|Parameter|Procedure|Constraint|Standard|Interface",
            "entity_name": "canonical name of the entity",
            "attributes": {{}},
            "evidence_text": "exact quote from the content",
            "confidence": 0.0-1.0
        }}
    ]
}}

Be precise and only extract entities with clear evidence."""

    RELATION_EXTRACTION_PROMPT = """Given the following entities extracted from a technical document,
identify relationships between them.

Entities:
{entities}

Original text:
{text}

{relation_types}

Output a JSON object:
{{
    "relations": [
        {{
            "source_entity": "entity name",
            "relation_type": "relation type from above",
            "target_entity": "entity name",
            "confidence": 0.0-1.0,
            "evidence": "text supporting this relation"
        }}
    ]
}}

Only include relations with clear evidence in the text."""

    IMAGE_EXTRACTION_PROMPT = """Analyze this technical image and extract structured entities.

Image context: {context}

{ontology}

Instructions:
1. Identify components, labels, values, and connections visible in the image
2. Extract technical parameters shown (numbers, units, specifications)
3. Note any component relationships shown in diagrams
4. Include confidence scores based on visibility and clarity

Output a JSON object:
{{
    "entities": [
        {{
            "entity_type": "Component|Parameter|Procedure|Constraint|Standard|Interface",
            "entity_name": "name from image",
            "attributes": {{}},
            "visual_description": "how it appears in the image",
            "location_in_image": "top-left, center, etc.",
            "confidence": 0.0-1.0
        }}
    ]
}}"""

    def __init__(self, config: Optional[EntityConfig] = None):
        self.config = config or get_config().entity
        self._llm_client = None
        self._multimodal_llm = None
        self.ontology = IndustrialOntology()

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    @property
    def multimodal_llm(self):
        if self._multimodal_llm is None:
            config = get_config().multimodal_llm
            self._multimodal_llm = LLMClient.get_client(config)
        return self._multimodal_llm

    def extract_from_text(
        self,
        text: str,
        source: SourceMetadata
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        从文本中抽取实体和关系

        Args:
            text: 文本内容
            source: 来源信息

        Returns:
            (实体列表, 关系列表)
        """
        # 构建prompt
        prompt = self.EXTRACTION_PROMPT.format(
            ontology=self.ontology.get_ontology_prompt(),
            content=text[:4000],  # 限制长度
            document=source.document,
            page=source.page,
            section=source.section or "N/A"
        )

        # 调用LLM
        try:
            response = self.llm_client.generate_structured(
                prompt,
                schema=self.ontology.get_entity_schema()
            )
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return [], []

        # 解析实体
        entities = []
        raw_entities = response.get("entities", [])

        for raw in raw_entities:
            try:
                # 过滤低置信度
                if raw.get("confidence", 0) < self.config.confidence_threshold:
                    continue

                entity = Entity(
                    entity_id=self._generate_entity_id(raw),
                    entity_type=EntityType(raw["entity_type"]),
                    entity_name=raw["entity_name"],
                    attributes=raw.get("attributes", {}),
                    source=source,
                    evidence_text=raw.get("evidence_text", ""),
                    confidence=raw.get("confidence", 0.5),
                    extraction_modality=ModalityType.TEXT
                )
                entities.append(entity)

            except Exception as e:
                self.logger.warning(f"Failed to parse entity: {e}")
                continue

        # 抽取关系
        relations = []
        if self.config.extract_relations and entities:
            relations = self._extract_relations(entities, text)

        self.logger.debug(f"Extracted {len(entities)} entities, {len(relations)} relations from text")
        return entities, relations

    def extract_from_image(
        self,
        image_path: str,
        source: SourceMetadata,
        context: Optional[str] = None
    ) -> List[Entity]:
        """
        从图像中抽取实体

        Args:
            image_path: 图像路径
            source: 来源信息
            context: 周围文本上下文

        Returns:
            实体列表
        """
        if not Path(image_path).exists():
            self.logger.warning(f"Image not found: {image_path}")
            return []

        # 构建prompt
        prompt = self.IMAGE_EXTRACTION_PROMPT.format(
            context=context[:1000] if context else "No context provided",
            ontology=self.ontology.get_ontology_prompt()
        )

        # 调用多模态LLM
        try:
            response_text = self.multimodal_llm.generate_with_image(prompt, image_path)
            # 解析JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                response = json.loads(response_text[start:end])
            else:
                response = {"entities": []}
        except Exception as e:
            self.logger.error(f"Image entity extraction failed: {e}")
            return []

        # 解析实体
        entities = []
        for raw in response.get("entities", []):
            try:
                if raw.get("confidence", 0) < self.config.confidence_threshold:
                    continue

                entity = Entity(
                    entity_id=self._generate_entity_id(raw),
                    entity_type=EntityType(raw["entity_type"]),
                    entity_name=raw["entity_name"],
                    attributes={
                        **raw.get("attributes", {}),
                        "visual_description": raw.get("visual_description", ""),
                        "location_in_image": raw.get("location_in_image", "")
                    },
                    source=source,
                    evidence_text=raw.get("visual_description", ""),
                    confidence=raw.get("confidence", 0.5),
                    extraction_modality=ModalityType.IMAGE
                )
                entities.append(entity)

            except Exception as e:
                self.logger.warning(f"Failed to parse image entity: {e}")
                continue

        self.logger.debug(f"Extracted {len(entities)} entities from image")
        return entities

    def extract_from_context(
        self,
        text_context: str,
        image_paths: List[str],
        source: SourceMetadata
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        从多模态上下文中抽取实体

        Args:
            text_context: 文本上下文
            image_paths: 图像路径列表
            source: 来源信息

        Returns:
            (实体列表, 关系列表)
        """
        all_entities = []
        all_relations = []

        # 文本实体抽取
        if text_context:
            text_entities, text_relations = self.extract_from_text(text_context, source)
            all_entities.extend(text_entities)
            all_relations.extend(text_relations)

        # 图像实体抽取
        for image_path in image_paths:
            image_entities = self.extract_from_image(
                image_path,
                source,
                context=text_context[:500]
            )
            all_entities.extend(image_entities)

        # 实体融合去重
        merged_entities = self._merge_entities(all_entities)

        # 跨模态关系抽取
        if len(merged_entities) > 1:
            cross_relations = self._extract_cross_modal_relations(
                merged_entities,
                text_context
            )
            all_relations.extend(cross_relations)

        return merged_entities, all_relations

    def _extract_relations(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Relation]:
        """抽取实体间关系"""
        if len(entities) < 2:
            return []

        # 构建实体列表文本
        entity_list = "\n".join([
            f"- {e.entity_name} ({e.entity_type.value})"
            for e in entities
        ])

        # 关系类型描述
        relation_types = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.ontology.RELATION_TYPES.items()
        ])

        prompt = self.RELATION_EXTRACTION_PROMPT.format(
            entities=entity_list,
            text=text[:3000],
            relation_types=relation_types
        )

        try:
            response = self.llm_client.generate_structured(
                prompt,
                schema=self.ontology.get_relation_schema()
            )
        except Exception as e:
            self.logger.warning(f"Relation extraction failed: {e}")
            return []

        relations = []
        for raw in response.get("relations", []):
            try:
                relation = Relation(
                    relation_id=self._generate_relation_id(raw),
                    source_entity=raw["source_entity"],
                    relation_type=raw["relation_type"],
                    target_entity=raw["target_entity"],
                    confidence=raw.get("confidence", 0.5),
                    evidence=raw.get("evidence", "")
                )
                relations.append(relation)
            except Exception as e:
                self.logger.warning(f"Failed to parse relation: {e}")

        return relations

    def _extract_cross_modal_relations(
        self,
        entities: List[Entity],
        text_context: str
    ) -> List[Relation]:
        """抽取跨模态关系（文本实体与图像实体之间）"""
        text_entities = [e for e in entities if e.extraction_modality == ModalityType.TEXT]
        image_entities = [e for e in entities if e.extraction_modality == ModalityType.IMAGE]

        if not text_entities or not image_entities:
            return []

        # 简化实现：基于名称相似度匹配
        relations = []
        for text_entity in text_entities:
            for image_entity in image_entities:
                # 名称相似度匹配
                if self._is_similar_name(text_entity.entity_name, image_entity.entity_name):
                    relation = Relation(
                        relation_id=f"cross_{text_entity.entity_id}_{image_entity.entity_id}",
                        source_entity=text_entity.entity_name,
                        relation_type="related_to",
                        target_entity=image_entity.entity_name,
                        confidence=0.7,
                        evidence="Cross-modal entity alignment"
                    )
                    relations.append(relation)

        return relations

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体融合去重"""
        if not entities:
            return []

        # 基于名称的简单去重
        seen = {}
        merged = []

        for entity in entities:
            key = entity.entity_name.lower().strip()

            if key not in seen:
                seen[key] = entity
                merged.append(entity)
            else:
                # 合并属性，保留置信度更高的
                existing = seen[key]
                if entity.confidence > existing.confidence:
                    # 合并属性
                    existing.attributes.update(entity.attributes)
                    existing.confidence = entity.confidence
                    # 如果一个是文本一个是图像，标记为混合
                    if existing.extraction_modality != entity.extraction_modality:
                        existing.extraction_modality = ModalityType.MIXED

        return merged

    def _is_similar_name(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """判断两个名称是否相似"""
        # 简单实现：规范化后比较
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        if n1 == n2:
            return True

        # 检查包含关系
        if n1 in n2 or n2 in n1:
            return True

        # 可以添加更复杂的相似度计算
        return False

    def _generate_entity_id(self, raw_entity: Dict) -> str:
        """生成实体ID"""
        content = f"{raw_entity.get('entity_type', '')}_{raw_entity.get('entity_name', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_relation_id(self, raw_relation: Dict) -> str:
        """生成关系ID"""
        content = f"{raw_relation.get('source_entity', '')}_{raw_relation.get('relation_type', '')}_{raw_relation.get('target_entity', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
