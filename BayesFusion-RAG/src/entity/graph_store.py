"""
Neo4j图数据库存储模块（可选）
提供更强大的图查询能力
"""

from typing import List, Dict, Any, Optional

from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.data_types import Entity, Relation


class EntityKnowledgeGraph(LoggerMixin):
    """
    实体知识图谱存储（Neo4j）
    提供图查询和多跳推理能力

    注意：此模块需要安装neo4j驱动和运行Neo4j服务
    pip install neo4j
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        config = get_config()
        neo4j_config = config.get("entity", "neo4j", default={})

        self.uri = uri or neo4j_config.get("uri", "bolt://localhost:7687")
        self.user = user or neo4j_config.get("user", "neo4j")
        self.password = password or neo4j_config.get("password", "")

        self._driver = None
        self._connected = False

    def connect(self) -> bool:
        """连接到Neo4j"""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 测试连接
            with self._driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            self.logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except ImportError:
            self.logger.error("neo4j package not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._connected = False

    def add_entity(self, entity: Entity) -> bool:
        """添加实体到图数据库"""
        if not self._connected:
            self.logger.warning("Not connected to Neo4j")
            return False

        entity_type = entity.entity_type.value
        query = f"""
        MERGE (e:{entity_type} {{entity_id: $entity_id}})
        SET e.name = $name,
            e.confidence = $confidence,
            e += $attributes
        MERGE (s:Source {{
            document: $document,
            page: $page,
            section: $section
        }})
        MERGE (e)-[:DEFINED_IN]->(s)
        """

        try:
            with self._driver.session() as session:
                session.run(query,
                    entity_id=entity.entity_id,
                    name=entity.entity_name,
                    confidence=entity.confidence,
                    attributes=entity.attributes,
                    document=entity.source.document,
                    page=entity.source.page,
                    section=entity.source.section or ""
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add entity: {e}")
            return False

    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        if not self._connected:
            return False

        # 动态创建关系类型
        rel_type = relation.relation_type.upper()
        query = f"""
        MATCH (s {{name: $source}})
        MATCH (t {{name: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.confidence = $confidence,
            r.evidence = $evidence
        """

        try:
            with self._driver.session() as session:
                session.run(query,
                    source=relation.source_entity,
                    target=relation.target_entity,
                    confidence=relation.confidence,
                    evidence=relation.evidence or ""
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add relation: {e}")
            return False

    def query_entity_context(
        self,
        entity_name: str,
        hops: int = 2
    ) -> Dict[str, Any]:
        """
        查询实体及其N跳邻居
        用于增强RAG上下文

        Args:
            entity_name: 实体名称
            hops: 跳数

        Returns:
            子图信息
        """
        if not self._connected:
            return {}

        query = f"""
        MATCH path = (e {{name: $name}})-[*1..{hops}]-(related)
        RETURN e, path, related
        LIMIT 50
        """

        try:
            with self._driver.session() as session:
                result = session.run(query, name=entity_name)
                records = list(result)

            # 解析结果
            entities = set()
            relations = []

            for record in records:
                if record.get("e"):
                    entities.add(record["e"]["name"])
                if record.get("related"):
                    entities.add(record["related"]["name"])

            return {
                "center_entity": entity_name,
                "connected_entities": list(entities),
                "hop_depth": hops,
                "total_nodes": len(entities)
            }

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {}

    def get_entity_by_type(self, entity_type: str, limit: int = 100) -> List[Dict]:
        """按类型查询实体"""
        if not self._connected:
            return []

        query = f"""
        MATCH (e:{entity_type})
        RETURN e
        LIMIT $limit
        """

        try:
            with self._driver.session() as session:
                result = session.run(query, limit=limit)
                return [dict(record["e"]) for record in result]
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return []

    def find_path(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 4
    ) -> List[Dict]:
        """
        查找两个实体间的路径

        Args:
            source_name: 起始实体名称
            target_name: 目标实体名称
            max_hops: 最大跳数

        Returns:
            路径列表
        """
        if not self._connected:
            return []

        query = f"""
        MATCH path = shortestPath(
            (s {{name: $source}})-[*1..{max_hops}]-(t {{name: $target}})
        )
        RETURN path
        """

        try:
            with self._driver.session() as session:
                result = session.run(query, source=source_name, target=target_name)
                paths = []
                for record in result:
                    path = record["path"]
                    nodes = [node["name"] for node in path.nodes]
                    rels = [rel.type for rel in path.relationships]
                    paths.append({
                        "nodes": nodes,
                        "relations": rels,
                        "length": len(rels)
                    })
                return paths
        except Exception as e:
            self.logger.error(f"Path query failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        if not self._connected:
            return {}

        try:
            with self._driver.session() as session:
                # 节点数量
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]

                # 关系数量
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]

                # 各类型节点数量
                type_result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as types, count(*) as count
                """)
                type_counts = {str(r["types"]): r["count"] for r in type_result}

            return {
                "total_nodes": node_count,
                "total_relations": rel_count,
                "nodes_by_type": type_counts,
                "connected": True
            }
        except Exception as e:
            self.logger.error(f"Statistics query failed: {e}")
            return {"connected": False, "error": str(e)}

    def clear(self) -> bool:
        """清空图数据库"""
        if not self._connected:
            return False

        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Graph database cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")
            return False
