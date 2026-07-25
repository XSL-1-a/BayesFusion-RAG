"""
MoRE (Mixture-of-Retrieval-Experts) 特征提取器
从查询和候选文档中提取用于门控网络的特征向量

特征组成:
- 查询嵌入 (query_dim, 默认3072)
- 实体特征 (6维): has_component, has_symptom, has_action, num_entities, avg_idf, entity_score_std
- 统计特征 (5维): top1_score, score_std, score_gap, score_entropy, top5_mean
- 查询类型 one-hot (4维)
- 查询长度特征 (3维): num_tokens, avg_token_len, has_question_word

总计: query_dim + 18

作者: RAG Research Team
日期: 2026-01-28
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field


@dataclass
class MoREFeatures:
    """MoRE 门控网络输入特征"""
    query_embedding: np.ndarray     # query_dim 维
    # 实体特征 (6维)
    has_component: float            # 0/1
    has_symptom: float              # 0/1
    has_action: float               # 0/1
    num_matched_entities: float     # 归一化到 [0,1]
    avg_entity_idf: float
    entity_score_std: float
    # 检索分数统计特征 (5维)
    top1_score: float
    score_std: float
    score_gap: float                # top1 - mean
    score_entropy: float
    top5_mean: float
    # 查询类型 (4类)
    query_type: int                 # 0-3
    # 查询长度特征 (3维)
    num_tokens: float               # 归一化
    avg_token_len: float
    has_question_word: float        # 0/1


class MoREFeatureExtractor:
    """
    MoRE 特征提取器
    从查询文本和候选文档中提取用于门控网络的特征向量

    与 ECGFeatureExtractor 共享实体提取逻辑，但增加了查询长度特征
    """

    QUERY_TYPE_MAP = {
        'component_symptom': 0,
        'procedure': 1,
        'specification': 2,
        'general': 3
    }

    COMPONENTS = {
        'engine', 'propeller', 'magneto', 'carburetor', 'cylinder', 'piston',
        'valve', 'gasket', 'fuel pump', 'oil pump', 'throttle', 'alternator',
        'battery', 'fuel line', 'oil filter', 'brake', 'tire', 'wheel',
        'hose', 'seal', 'bearing', 'transmission', 'radiator', 'hvac',
        'compressor', 'pump', 'motor', 'pipe', 'drain', 'faucet'
    }

    SYMPTOMS = {
        'leaking', 'cracked', 'worn', 'broken', 'failed', 'noisy', 'loose',
        'corroded', 'overheating', 'damaged', 'missing', 'stuck', 'seized'
    }

    ACTIONS = {
        'replace', 'replaced', 'repair', 'install', 'remove', 'inspect',
        'adjust', 'clean', 'tighten', 'test', 'check', 'calibrate'
    }

    QUESTION_WORDS = {
        'how', 'what', 'where', 'when', 'why', 'which', 'who',
        'can', 'does', 'is', 'are', 'should'
    }

    def __init__(self, embedder=None, entity_kb=None, query_dim: int = 3072):
        """
        Args:
            embedder: 文本嵌入器，需要有 embed_text 方法
            entity_kb: 实体知识库 (可选)
            query_dim: 查询嵌入维度
        """
        self.embedder = embedder
        self.entity_kb = entity_kb
        self.query_dim = query_dim

    @property
    def feature_dim(self) -> int:
        """总特征维度"""
        return self.query_dim + 18  # 6 entity + 5 score + 4 type + 3 query

    def extract_features(
        self,
        query: str,
        candidates: Optional[List[Dict]] = None
    ) -> MoREFeatures:
        """
        提取查询特征

        Args:
            query: 查询文本
            candidates: 候选文档列表 (可选，用于统计特征)

        Returns:
            MoREFeatures 对象
        """
        # 查询嵌入
        if self.embedder is not None:
            query_embedding = np.array(
                self.embedder.embed_text(query), dtype=np.float32
            )
        else:
            np.random.seed(hash(query) % (2**32 - 1))
            query_embedding = np.random.randn(self.query_dim).astype(np.float32)
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # 实体特征
        q_comps, q_acts, q_syms = self._extract_from_query(query)
        has_component = 1.0 if q_comps else 0.0
        has_symptom = 1.0 if q_syms else 0.0
        has_action = 1.0 if q_acts else 0.0
        num_entities = len(q_comps | q_acts | q_syms)

        avg_idf = 0.0
        entity_score_std = 0.0
        if self.entity_kb is not None and num_entities > 0:
            all_entities = q_comps | q_acts | q_syms
            avg_idf = sum(
                self.entity_kb.get_idf(e) for e in all_entities
            ) / num_entities

            if candidates:
                entity_scores = []
                for r in candidates:
                    doc_comps, doc_acts, doc_syms = self.entity_kb.get_doc_entities(
                        r.get('doc_id', '')
                    )
                    score = self._compute_entity_score(
                        q_comps, q_acts, q_syms,
                        doc_comps, doc_acts, doc_syms
                    )
                    entity_scores.append(score)
                if entity_scores:
                    max_es = max(entity_scores) if max(entity_scores) > 0 else 1.0
                    normed = [es / max_es for es in entity_scores]
                    entity_score_std = float(np.std(normed))

        # 检索分数统计特征
        if candidates:
            scores = [r.get('score', 0) for r in candidates]
            if scores:
                score_std = float(np.std(scores))
                score_gap = scores[0] - float(np.mean(scores))
                probs = np.array(scores) / (sum(scores) + 1e-10)
                score_entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                top1_score = scores[0]
                top5_mean = float(np.mean(scores[:5]))
            else:
                score_std = score_gap = score_entropy = top1_score = top5_mean = 0.0
        else:
            score_std = score_gap = score_entropy = top1_score = top5_mean = 0.0

        # 查询类型
        query_type = self.QUERY_TYPE_MAP[self.classify_query_type(query)]

        # 查询长度特征
        tokens = query.split()
        num_tokens = min(len(tokens) / 20.0, 1.0)  # 归一化到 [0,1]
        avg_token_len = np.mean([len(t) for t in tokens]) / 10.0 if tokens else 0.0
        has_question_word = 1.0 if any(
            t.lower() in self.QUESTION_WORDS for t in tokens
        ) else 0.0

        return MoREFeatures(
            query_embedding=query_embedding,
            has_component=has_component,
            has_symptom=has_symptom,
            has_action=has_action,
            num_matched_entities=min(num_entities / 10.0, 1.0),
            avg_entity_idf=avg_idf,
            entity_score_std=entity_score_std,
            top1_score=top1_score,
            score_std=score_std,
            score_gap=score_gap,
            score_entropy=score_entropy,
            top5_mean=top5_mean,
            query_type=query_type,
            num_tokens=num_tokens,
            avg_token_len=avg_token_len,
            has_question_word=has_question_word
        )

    def to_tensor(self, features: MoREFeatures):
        """将特征转换为 PyTorch Tensor"""
        import torch

        query_type_onehot = np.zeros(4, dtype=np.float32)
        query_type_onehot[features.query_type] = 1.0

        feature_vector = np.concatenate([
            features.query_embedding,
            # 实体特征 (6维)
            np.array([
                features.has_component, features.has_symptom, features.has_action,
                features.num_matched_entities, features.avg_entity_idf,
                features.entity_score_std
            ], dtype=np.float32),
            # 统计特征 (5维)
            np.array([
                features.top1_score, features.score_std, features.score_gap,
                features.score_entropy, features.top5_mean
            ], dtype=np.float32),
            # 查询类型 one-hot (4维)
            query_type_onehot,
            # 查询长度特征 (3维)
            np.array([
                features.num_tokens, features.avg_token_len,
                features.has_question_word
            ], dtype=np.float32)
        ])

        return torch.from_numpy(feature_vector)

    def classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        q = query.lower()
        q_comps, q_acts, q_syms = self._extract_from_query(query)
        if q_comps and q_syms:
            return 'component_symptom'
        elif q_acts or any(kw in q for kw in ['how', 'step', 'procedure']):
            return 'procedure'
        elif any(kw in q for kw in ['model', 'part', 'number', 'specification']):
            return 'specification'
        return 'general'

    def _extract_from_query(self, query: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """从查询中提取实体"""
        q = query.lower()
        comps = {c for c in self.COMPONENTS if c in q}
        acts = {a for a in self.ACTIONS if a in q}
        syms = {s for s in self.SYMPTOMS if s in q}
        return comps, acts, syms

    def _compute_entity_score(
        self,
        q_comps: Set, q_acts: Set, q_syms: Set,
        doc_comps: Set, doc_acts: Set, doc_syms: Set
    ) -> float:
        """计算实体匹配分数"""
        score = 0.0
        for c in q_comps & doc_comps:
            score += self.entity_kb.get_idf(c) * 2.0
        for a in q_acts & doc_acts:
            score += self.entity_kb.get_idf(a) * 1.0
        for s in q_syms & doc_syms:
            score += self.entity_kb.get_idf(s) * 1.5
        if (q_comps & doc_comps) and (q_syms & doc_syms):
            score *= 1.5
        return score
