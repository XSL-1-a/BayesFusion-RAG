"""
实体增强型RAG模块
将实体信息整合到RAG流程中
"""

from typing import List, Dict, Any, Optional, Tuple

from ..utils.logger import LoggerMixin
from ..utils.data_types import (
    Entity, RetrievalResult, SourceMetadata, ModalityType, RetrievalLevel
)
from .extractor import MultimodalEntityExtractor
from .store import LightweightEntityStore
from .ontology import IndustrialOntology


class EntityEnhancedRAG(LoggerMixin):
    """
    实体增强型RAG
    在检索和生成过程中利用实体信息

    功能：
    1. 查询实体识别：从查询中识别相关实体
    2. 实体上下文增强：添加实体相关信息到上下文
    3. 实体验证：验证答案中的实体准确性
    """

    def __init__(
        self,
        extractor: Optional[MultimodalEntityExtractor] = None,
        store: Optional[LightweightEntityStore] = None
    ):
        self.extractor = extractor or MultimodalEntityExtractor()
        self.store = store or LightweightEntityStore()
        self.ontology = IndustrialOntology()

    # 实体抽取时排除的停用词
    _STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
        'through', 'during', 'before', 'after', 'and', 'but', 'or', 'not',
        'no', 'as', 'if', 'it', 'its', 'this', 'that', 'these', 'those',
        'such', 'than', 'what', 'how', 'which', 'when', 'where', 'who',
        'why', 'there', 'here', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'any', 'new', 'used', 'also',
        'between', 'while', 'another', 'spare',
    }

    def _is_valid_entity(self, entity_name: str, min_confidence: float = 0.8) -> bool:
        """检查实体名称是否有效"""
        if not entity_name or len(entity_name) < 3:
            return False
        if '\n' in entity_name or '\r' in entity_name:
            return False
        cleaned = entity_name.strip().lower()
        if cleaned in self._STOP_WORDS:
            return False
        if cleaned.isdigit() or len(cleaned) <= 2:
            return False
        # 过滤以冠词/介词开头的噪声实体
        for prefix in ('a ', 'an ', 'the ', 'of ', 'in ', 'on ', 'at ', 'by '):
            if cleaned.startswith(prefix):
                return False
        return True

    def extract_query_entities(self, query: str) -> List[str]:
        """
        从查询中识别实体

        Args:
            query: 用户查询

        Returns:
            实体名称列表
        """
        entities = []
        seen = set()

        words = query.split()
        # 搜索单词和二元组
        candidates = []
        for word in words:
            w = word.strip('.,;:!?()[]"\'').lower()
            if len(w) > 3 and w not in self._STOP_WORDS:
                candidates.append(word.strip('.,;:!?()[]"\''))
        for i in range(len(words) - 1):
            bigram = f"{words[i].strip('.,;:!?()')} {words[i+1].strip('.,;:!?()')}"
            if len(bigram) > 5:
                candidates.append(bigram)

        for term in candidates:
            matches = self.store.search_entities(term, limit=3)
            for match in matches:
                name = match.get("entity_name", "")
                confidence = match.get("confidence", 0)
                if (name not in seen
                        and self._is_valid_entity(name)
                        and confidence >= 0.8):
                    entities.append(name)
                    seen.add(name)

        return entities[:8]

    def enhance_context_with_entities(
        self,
        retrieval_results: List[RetrievalResult],
        query_entities: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        用实体信息增强检索结果

        Args:
            retrieval_results: 原始检索结果
            query_entities: 查询中识别的实体

        Returns:
            增强后的检索结果
        """
        enhanced_results = []

        for result in retrieval_results:
            # 获取文档相关的实体
            doc_id = result.source.document
            doc_entities = self.store.query_by_document(doc_id)

            # 添加实体信息到元数据
            entity_info = []
            for entity in doc_entities[:5]:  # 限制数量
                entity_info.append({
                    "name": entity.get("entity_name"),
                    "type": entity.get("entity_type"),
                    "confidence": entity.get("confidence")
                })

            result.metadata["related_entities"] = entity_info

            # 如果有查询实体匹配，提升分数
            if query_entities:
                for entity in doc_entities:
                    if entity.get("entity_name") in query_entities:
                        result.score *= 1.2  # 提升20%
                        result.metadata["entity_match"] = True
                        break

            enhanced_results.append(result)

        # 重新排序
        enhanced_results.sort(key=lambda x: x.score, reverse=True)

        return enhanced_results

    def get_entity_context(
        self,
        entity_names: List[str],
        include_relations: bool = True
    ) -> str:
        """
        获取实体的上下文信息

        Args:
            entity_names: 实体名称列表
            include_relations: 是否包含关系信息

        Returns:
            实体上下文文本
        """
        context_parts = []

        for name in entity_names:
            entity_info = self.store.get_entity_with_source(name)
            if entity_info:
                entity = entity_info["entity"]
                context_parts.append(
                    f"Entity: {entity.get('entity_name')} ({entity.get('entity_type')})\n"
                    f"  Source: {entity.get('source', {}).get('document', 'Unknown')}, "
                    f"Page {entity.get('source', {}).get('page', 'N/A')}\n"
                    f"  Evidence: {entity.get('evidence_text', 'N/A')[:200]}"
                )

                # 添加关系信息
                if include_relations:
                    relations = self.store.get_entity_relations(name)
                    if relations:
                        rel_text = []
                        for rel in relations[:3]:  # 限制数量
                            rel_text.append(
                                f"    - {rel.get('relation_type')}: {rel.get('target_entity')}"
                            )
                        context_parts.append("  Relations:\n" + "\n".join(rel_text))

        return "\n\n".join(context_parts)

    def build_entity_enhanced_prompt(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        query_entities: Optional[List[str]] = None
    ) -> str:
        """
        构建实体增强的提示

        Args:
            query: 用户查询
            retrieval_results: 检索结果
            query_entities: 查询实体

        Returns:
            增强后的提示文本
        """
        parts = []

        # 实体上下文
        if query_entities:
            entity_context = self.get_entity_context(query_entities)
            if entity_context:
                parts.append("=== Relevant Entities ===\n" + entity_context)

        # 检索结果
        parts.append("\n=== Retrieved Context ===")
        for i, result in enumerate(retrieval_results, 1):
            source = result.source
            parts.append(
                f"\n[Source {i}] Document: {source.document}, "
                f"Page: {source.page}, Section: {source.section or 'N/A'}\n"
                f"{result.content[:1000]}"
            )

            # 添加相关实体
            related_entities = result.metadata.get("related_entities", [])
            if related_entities:
                entity_list = ", ".join([
                    f"{e['name']} ({e['type']})"
                    for e in related_entities[:3]
                ])
                parts.append(f"  Related entities: {entity_list}")

        return "\n".join(parts)

    def verify_answer_entities(
        self,
        answer: str,
        cited_sources: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        验证答案中提到的实体

        Args:
            answer: 生成的答案
            cited_sources: 引用的来源

        Returns:
            验证结果
        """
        # 简单实现：检查答案中提到的实体是否在知识库中
        verification_results = {
            "verified_entities": [],
            "unverified_entities": [],
            "overall_confidence": 0.0
        }

        # 搜索答案中可能提到的实体
        words = answer.split()
        for i, word in enumerate(words):
            if len(word) > 3:
                # 尝试匹配2-3个词的短语
                phrase = " ".join(words[i:i+2])
                matches = self.store.search_entities(phrase, limit=1)
                if matches:
                    verification_results["verified_entities"].append({
                        "phrase": phrase,
                        "matched_entity": matches[0].get("entity_name"),
                        "confidence": matches[0].get("confidence", 0)
                    })

        # 计算整体置信度
        if verification_results["verified_entities"]:
            confidences = [
                e["confidence"]
                for e in verification_results["verified_entities"]
            ]
            verification_results["overall_confidence"] = sum(confidences) / len(confidences)

        return verification_results

    def process_document_for_entities(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        images: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, int]:
        """
        处理文档并抽取实体

        Args:
            doc_id: 文档ID
            chunks: 文本块列表
            images: 图像列表

        Returns:
            (实体数量, 关系数量)
        """
        total_entities = 0
        total_relations = 0

        for chunk in chunks:
            source = SourceMetadata(
                document=doc_id,
                page=chunk.get("page", 0),
                section=chunk.get("section_title", "")
            )

            entities, relations = self.extractor.extract_from_text(
                chunk.get("content", ""),
                source
            )

            self.store.add_entities_batch(entities)
            self.store.add_relations_batch(relations)

            total_entities += len(entities)
            total_relations += len(relations)

        # 处理图像
        if images:
            for image in images:
                source = SourceMetadata(
                    document=doc_id,
                    page=image.get("page", 0),
                    section=image.get("section_id", "")
                )

                image_entities = self.extractor.extract_from_image(
                    image.get("image_path", ""),
                    source,
                    context=image.get("caption", "")
                )

                self.store.add_entities_batch(image_entities)
                total_entities += len(image_entities)

        self.logger.info(
            f"Processed document {doc_id}: "
            f"{total_entities} entities, {total_relations} relations"
        )

        return total_entities, total_relations
