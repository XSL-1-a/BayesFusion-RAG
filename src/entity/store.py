"""
轻量级实体存储模块
使用JSON + SQLite实现，无需外部数据库依赖
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..utils.data_types import Entity, Relation, EntityType


class LightweightEntityStore(LoggerMixin):
    """
    轻量级实体存储
    使用JSON文件存储实体，SQLite索引支持快速查询

    优势：
    - 无需外部数据库
    - 易于调试和版本控制
    - 支持导出用于设计核查
    """

    def __init__(self, storage_path: str = "./data/entity_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.entities_file = self.storage_path / "entities.json"
        self.relations_file = self.storage_path / "relations.json"
        self.index_db = self.storage_path / "entity_index.db"

        self._entities_cache: Optional[List[Dict]] = None
        self._relations_cache: Optional[List[Dict]] = None
        self._init_index()
        self._ensure_index_populated()

    def _init_index(self) -> None:
        """初始化SQLite索引"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()

        # 实体索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_index (
                entity_id TEXT PRIMARY KEY,
                entity_name TEXT,
                entity_type TEXT,
                doc_id TEXT,
                section TEXT,
                page INTEGER,
                confidence REAL,
                created_at TEXT
            )
        """)

        # 关系索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_index (
                relation_id TEXT PRIMARY KEY,
                source_entity TEXT,
                relation_type TEXT,
                target_entity TEXT,
                confidence REAL
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entity_index(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_doc ON entity_index(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_index(entity_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_source ON relation_index(source_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_target ON relation_index(target_entity)")

        conn.commit()
        conn.close()

    def _ensure_index_populated(self) -> None:
        """如果 SQLite 索引为空但 JSON 有数据，则自动重建索引"""
        conn = sqlite3.connect(self.index_db)
        count = conn.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0]
        conn.close()

        if count == 0:
            entities = self._load_entities()
            if entities:
                self.logger.info(f"Rebuilding SQLite index from {len(entities)} entities...")
                conn = sqlite3.connect(self.index_db)
                for e in entities:
                    # 兼容扁平和嵌套两种字段格式
                    doc_id = (e.get("source", {}).get("document", "")
                              if isinstance(e.get("source"), dict)
                              else e.get("source_document", ""))
                    section = (e.get("source", {}).get("section", "")
                               if isinstance(e.get("source"), dict)
                               else e.get("source_section", ""))
                    page = (e.get("source", {}).get("page", 0)
                            if isinstance(e.get("source"), dict)
                            else e.get("source_page", 0))
                    conn.execute("""
                        INSERT OR REPLACE INTO entity_index
                        (entity_id, entity_name, entity_type, doc_id, section, page, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        e.get("entity_id", ""),
                        e.get("entity_name", ""),
                        e.get("entity_type", ""),
                        doc_id, section, page,
                        e.get("confidence", 0),
                        e.get("created_at", "")
                    ))
                conn.commit()
                conn.close()
                self.logger.info(f"SQLite index rebuilt: {len(entities)} entities indexed")

                # 同样重建关系索引
                relations = self._load_relations()
                if relations:
                    conn = sqlite3.connect(self.index_db)
                    for r in relations:
                        conn.execute("""
                            INSERT OR REPLACE INTO relation_index
                            (relation_id, source_entity, relation_type, target_entity, confidence)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            r.get("relation_id", ""),
                            r.get("source_entity", ""),
                            r.get("relation_type", ""),
                            r.get("target_entity", ""),
                            r.get("confidence", 0)
                        ))
                    conn.commit()
                    conn.close()
                    self.logger.info(f"Relation index rebuilt: {len(relations)} relations indexed")

    def _load_entities(self) -> List[Dict]:
        """加载实体数据"""
        if self._entities_cache is not None:
            return self._entities_cache

        if self.entities_file.exists():
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                self._entities_cache = json.load(f)
        else:
            self._entities_cache = []

        return self._entities_cache

    def _save_entities(self, entities: List[Dict]) -> None:
        """保存实体数据"""
        with open(self.entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        self._entities_cache = entities

    def _load_relations(self) -> List[Dict]:
        """加载关系数据"""
        if self._relations_cache is not None:
            return self._relations_cache

        if self.relations_file.exists():
            with open(self.relations_file, 'r', encoding='utf-8') as f:
                self._relations_cache = json.load(f)
        else:
            self._relations_cache = []

        return self._relations_cache

    def _save_relations(self, relations: List[Dict]) -> None:
        """保存关系数据"""
        with open(self.relations_file, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        self._relations_cache = relations

    def add_entity(self, entity: Entity) -> str:
        """
        添加实体

        Args:
            entity: Entity对象

        Returns:
            实体ID
        """
        entities = self._load_entities()

        # 转换为字典
        entity_dict = entity.to_dict()
        entity_dict["created_at"] = datetime.now().isoformat()

        # 检查是否已存在
        existing_idx = None
        for i, e in enumerate(entities):
            if e.get("entity_id") == entity.entity_id:
                existing_idx = i
                break

        if existing_idx is not None:
            # 更新已存在的实体
            entities[existing_idx] = entity_dict
        else:
            # 添加新实体
            entities.append(entity_dict)

        # 保存JSON
        self._save_entities(entities)

        # 更新索引
        self._update_entity_index(entity_dict)

        return entity.entity_id

    def add_entities_batch(self, entities: List[Entity]) -> List[str]:
        """批量添加实体"""
        ids = []
        for entity in entities:
            entity_id = self.add_entity(entity)
            ids.append(entity_id)
        return ids

    def add_relation(self, relation: Relation) -> str:
        """添加关系"""
        relations = self._load_relations()
        relation_dict = relation.to_dict()

        # 检查是否已存在
        existing = False
        for i, r in enumerate(relations):
            if r.get("relation_id") == relation.relation_id:
                relations[i] = relation_dict
                existing = True
                break

        if not existing:
            relations.append(relation_dict)

        self._save_relations(relations)
        self._update_relation_index(relation_dict)

        return relation.relation_id

    def add_relations_batch(self, relations: List[Relation]) -> List[str]:
        """批量添加关系"""
        ids = []
        for relation in relations:
            relation_id = self.add_relation(relation)
            ids.append(relation_id)
        return ids

    def _update_entity_index(self, entity: Dict) -> None:
        """更新实体索引（兼容扁平和嵌套字段格式）"""
        doc_id = (entity.get("source", {}).get("document", "")
                  if isinstance(entity.get("source"), dict)
                  else entity.get("source_document", ""))
        section = (entity.get("source", {}).get("section", "")
                   if isinstance(entity.get("source"), dict)
                   else entity.get("source_section", ""))
        page = (entity.get("source", {}).get("page", 0)
                if isinstance(entity.get("source"), dict)
                else entity.get("source_page", 0))
        conn = sqlite3.connect(self.index_db)
        conn.execute("""
            INSERT OR REPLACE INTO entity_index
            (entity_id, entity_name, entity_type, doc_id, section, page, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.get("entity_id"),
            entity.get("entity_name"),
            entity.get("entity_type"),
            doc_id, section, page,
            entity.get("confidence", 0),
            entity.get("created_at", "")
        ))
        conn.commit()
        conn.close()

    def _update_relation_index(self, relation: Dict) -> None:
        """更新关系索引"""
        conn = sqlite3.connect(self.index_db)
        conn.execute("""
            INSERT OR REPLACE INTO relation_index
            (relation_id, source_entity, relation_type, target_entity, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (
            relation.get("relation_id"),
            relation.get("source_entity"),
            relation.get("relation_type"),
            relation.get("target_entity"),
            relation.get("confidence", 0)
        ))
        conn.commit()
        conn.close()

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """根据ID获取实体"""
        entities = self._load_entities()
        for e in entities:
            if e.get("entity_id") == entity_id:
                return e
        return None

    def get_entity_by_name(self, entity_name: str) -> Optional[Dict]:
        """根据名称获取实体"""
        entities = self._load_entities()
        for e in entities:
            if e.get("entity_name", "").lower() == entity_name.lower():
                return e
        return None

    def query_by_type(self, entity_type: str) -> List[Dict]:
        """按类型查询实体"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.execute(
            "SELECT entity_id FROM entity_index WHERE entity_type = ?",
            (entity_type,)
        )
        entity_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        entities = self._load_entities()
        return [e for e in entities if e.get("entity_id") in entity_ids]

    def query_by_document(self, doc_id: str) -> List[Dict]:
        """按文档查询实体"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.execute(
            "SELECT entity_id FROM entity_index WHERE doc_id = ?",
            (doc_id,)
        )
        entity_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        entities = self._load_entities()
        return [e for e in entities if e.get("entity_id") in entity_ids]

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """搜索实体"""
        conn = sqlite3.connect(self.index_db)

        if entity_type:
            cursor = conn.execute(
                """SELECT entity_id FROM entity_index
                   WHERE entity_name LIKE ? AND entity_type = ?
                   ORDER BY confidence DESC LIMIT ?""",
                (f"%{query}%", entity_type, limit)
            )
        else:
            cursor = conn.execute(
                """SELECT entity_id FROM entity_index
                   WHERE entity_name LIKE ?
                   ORDER BY confidence DESC LIMIT ?""",
                (f"%{query}%", limit)
            )

        entity_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        entities = self._load_entities()
        return [e for e in entities if e.get("entity_id") in entity_ids]

    def get_entity_relations(self, entity_name: str) -> List[Dict]:
        """获取实体的所有关系"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.execute(
            """SELECT relation_id FROM relation_index
               WHERE source_entity = ? OR target_entity = ?""",
            (entity_name, entity_name)
        )
        relation_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        relations = self._load_relations()
        return [r for r in relations if r.get("relation_id") in relation_ids]

    def get_entity_with_source(self, entity_name: str) -> Optional[Dict]:
        """获取实体及其完整来源信息（支持设计核查）"""
        entity = self.get_entity_by_name(entity_name)
        if entity:
            return {
                "entity": entity,
                "source": entity.get("source", {}),
                "evidence": entity.get("evidence_text", ""),
                "confidence": entity.get("confidence", 0),
                "traceable": True
            }
        return None

    def export_for_design_check(self) -> Dict[str, Any]:
        """导出用于设计核查的结构化数据"""
        entities = self._load_entities()
        relations = self._load_relations()

        return {
            "components": [e for e in entities if e.get("entity_type") == "Component"],
            "parameters": [e for e in entities if e.get("entity_type") == "Parameter"],
            "constraints": [e for e in entities if e.get("entity_type") == "Constraint"],
            "procedures": [e for e in entities if e.get("entity_type") == "Procedure"],
            "standards": [e for e in entities if e.get("entity_type") == "Standard"],
            "interfaces": [e for e in entities if e.get("entity_type") == "Interface"],
            "relations": relations,
            "total_entities": len(entities),
            "total_relations": len(relations),
            "documents_covered": list(set(
                e.get("source", {}).get("document", "")
                for e in entities
            )),
            "export_time": datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        entities = self._load_entities()
        relations = self._load_relations()

        type_counts = {}
        for e in entities:
            t = e.get("entity_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entities_by_type": type_counts,
            "unique_documents": len(set(
                e.get("source", {}).get("document", "")
                for e in entities
            )),
            "avg_confidence": sum(e.get("confidence", 0) for e in entities) / max(1, len(entities))
        }

    def clear(self) -> None:
        """清空存储"""
        self._save_entities([])
        self._save_relations([])

        conn = sqlite3.connect(self.index_db)
        conn.execute("DELETE FROM entity_index")
        conn.execute("DELETE FROM relation_index")
        conn.commit()
        conn.close()

        self._entities_cache = None
        self._relations_cache = None

        self.logger.info("Entity store cleared")
