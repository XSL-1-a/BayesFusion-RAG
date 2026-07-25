"""
MoRE (Mixture-of-Retrieval-Experts) 检索专家模块
定义统一的检索专家接口和四种专家实现

专家类型:
1. BM25Expert: 基于 BM25 的稀疏检索
2. DenseExpert: 基于稠密向量的语义检索
3. EntityExpert: 基于实体图谱的结构化检索
4. ImageExpert: 基于 CLIP 的多模态检索

每个专家接收查询和候选文档，返回统一格式的评分列表

作者: RAG Research Team
日期: 2026-01-28
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ExpertType(Enum):
    """专家类型枚举"""
    BM25 = "bm25"
    DENSE = "dense"
    ENTITY = "entity"
    IMAGE = "image"


@dataclass
class ExpertResult:
    """单个专家的检索结果"""
    expert_name: str
    expert_type: ExpertType
    scored_docs: List[Dict[str, Any]]  # [{doc_id, score, content, ...}]
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseExpert(ABC):
    """
    检索专家基类

    所有专家必须实现 score 方法，该方法:
    - 输入: 查询文本 + 候选文档列表
    - 输出: 对每个文档的相关性评分 (0~1归一化)
    """

    def __init__(self, name: str, expert_type: ExpertType):
        self.name = name
        self.expert_type = expert_type

    @abstractmethod
    def score(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 20
    ) -> ExpertResult:
        """
        对候选文档评分

        Args:
            query: 查询文本
            candidates: 候选文档列表，每个包含 doc_id, content, score 等
            top_k: 返回 top-k 结果

        Returns:
            ExpertResult 包含评分后的文档列表
        """
        pass

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """将分数归一化到 [0, 1]"""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s < 1e-10:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]


class BM25Expert(BaseExpert):
    """
    BM25 稀疏检索专家
    基于词频和逆文档频率的经典检索方法
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, name: str = "BM25"):
        super().__init__(name=name, expert_type=ExpertType.BM25)
        self.k1 = k1
        self.b = b
        self._idf_cache: Dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._doc_count: int = 0

    def build_index(self, documents: List[Dict]):
        """构建 BM25 索引 (IDF 缓存)"""
        self._doc_count = len(documents)
        doc_lens = []
        term_doc_freq = {}

        for doc in documents:
            content = doc.get('content', '').lower()
            tokens = content.split()
            doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

        self._avg_doc_len = np.mean(doc_lens) if doc_lens else 1.0

        for term, df in term_doc_freq.items():
            self._idf_cache[term] = np.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

    def score(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 20
    ) -> ExpertResult:
        """BM25 评分"""
        import time
        start = time.time()

        query_tokens = query.lower().split()
        raw_scores = []

        for doc in candidates:
            content = doc.get('content', '').lower()
            doc_tokens = content.split()
            doc_len = len(doc_tokens)

            score = 0.0
            for qt in query_tokens:
                tf = doc_tokens.count(qt)
                idf = self._idf_cache.get(qt, 0.0)
                if idf == 0.0 and self._doc_count == 0:
                    idf = 1.0  # 没有索引时给默认值

                numerator = tf * (self.k1 + 1)
                avg_dl = self._avg_doc_len if self._avg_doc_len > 0 else 1.0
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_dl)
                score += idf * numerator / (denominator + 1e-10)

            raw_scores.append(score)

        # 归一化
        normed = self._normalize_scores(raw_scores)

        scored_docs = []
        for doc, ns in zip(candidates, normed):
            scored_doc = doc.copy()
            scored_doc['expert_score'] = ns
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['expert_score'], reverse=True)
        latency = (time.time() - start) * 1000

        return ExpertResult(
            expert_name=self.name,
            expert_type=self.expert_type,
            scored_docs=scored_docs[:top_k],
            latency_ms=latency
        )


class DenseExpert(BaseExpert):
    """
    稠密向量检索专家
    基于预训练嵌入模型的语义相似度检索
    """

    def __init__(self, embedder=None, name: str = "Dense"):
        super().__init__(name=name, expert_type=ExpertType.DENSE)
        self.embedder = embedder

    def score(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 20
    ) -> ExpertResult:
        """稠密向量余弦相似度评分"""
        import time
        start = time.time()

        # 获取查询嵌入
        if self.embedder is not None:
            query_emb = np.array(self.embedder.embed_text(query), dtype=np.float32)
        else:
            # Mock: 使用确定性哈希生成
            np.random.seed(hash(query) % (2**32 - 1))
            query_emb = np.random.randn(512).astype(np.float32)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        raw_scores = []
        for doc in candidates:
            if 'embedding' in doc and doc['embedding'] is not None:
                doc_emb = np.array(doc['embedding'], dtype=np.float32)
            elif self.embedder is not None:
                doc_emb = np.array(
                    self.embedder.embed_text(doc.get('content', '')),
                    dtype=np.float32
                )
            else:
                np.random.seed(hash(doc.get('content', '')) % (2**32 - 1))
                doc_emb = np.random.randn(512).astype(np.float32)
                doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)

            # 余弦相似度
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-10
            )
            raw_scores.append(float(sim))

        normed = self._normalize_scores(raw_scores)

        scored_docs = []
        for doc, ns in zip(candidates, normed):
            scored_doc = doc.copy()
            scored_doc['expert_score'] = ns
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['expert_score'], reverse=True)
        latency = (time.time() - start) * 1000

        return ExpertResult(
            expert_name=self.name,
            expert_type=self.expert_type,
            scored_docs=scored_docs[:top_k],
            latency_ms=latency
        )


class EntityExpert(BaseExpert):
    """
    实体图谱检索专家
    基于查询和文档中实体匹配度的结构化检索
    复用 ECGFeatureExtractor 的实体提取逻辑
    """

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

    def __init__(self, entity_kb=None, name: str = "Entity"):
        super().__init__(name=name, expert_type=ExpertType.ENTITY)
        self.entity_kb = entity_kb

    def score(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 20
    ) -> ExpertResult:
        """实体匹配评分"""
        import time
        start = time.time()

        q = query.lower()
        q_comps = {c for c in self.COMPONENTS if c in q}
        q_acts = {a for a in self.ACTIONS if a in q}
        q_syms = {s for s in self.SYMPTOMS if s in q}

        raw_scores = []
        for doc in candidates:
            content = doc.get('content', '').lower()
            d_comps = {c for c in self.COMPONENTS if c in content}
            d_acts = {a for a in self.ACTIONS if a in content}
            d_syms = {s for s in self.SYMPTOMS if s in content}

            score = 0.0
            # 组件匹配 (最高权重)
            comp_overlap = len(q_comps & d_comps)
            score += comp_overlap * 3.0
            # 症状匹配
            sym_overlap = len(q_syms & d_syms)
            score += sym_overlap * 2.0
            # 动作匹配
            act_overlap = len(q_acts & d_acts)
            score += act_overlap * 1.0
            # 组件+症状共现加成
            if comp_overlap > 0 and sym_overlap > 0:
                score *= 1.5

            # 使用实体知识库 (如果可用)
            if self.entity_kb is not None:
                doc_id = doc.get('doc_id', '')
                try:
                    kb_comps, kb_acts, kb_syms = self.entity_kb.get_doc_entities(doc_id)
                    kb_score = 0.0
                    for c in q_comps & kb_comps:
                        kb_score += self.entity_kb.get_idf(c) * 2.0
                    for a in q_acts & kb_acts:
                        kb_score += self.entity_kb.get_idf(a) * 1.0
                    for s in q_syms & kb_syms:
                        kb_score += self.entity_kb.get_idf(s) * 1.5
                    score += kb_score
                except Exception:
                    pass

            raw_scores.append(score)

        normed = self._normalize_scores(raw_scores)

        scored_docs = []
        for doc, ns in zip(candidates, normed):
            scored_doc = doc.copy()
            scored_doc['expert_score'] = ns
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['expert_score'], reverse=True)
        latency = (time.time() - start) * 1000

        return ExpertResult(
            expert_name=self.name,
            expert_type=self.expert_type,
            scored_docs=scored_docs[:top_k],
            latency_ms=latency
        )


class ImageExpert(BaseExpert):
    """
    多模态图像检索专家
    基于 CLIP 或图像摘要的跨模态检索

    对纯文本候选，使用图像相关性信号 (标题/描述中的视觉词汇)
    """

    VISUAL_KEYWORDS = {
        'diagram', 'figure', 'photo', 'image', 'illustration', 'chart',
        'drawing', 'schematic', 'picture', 'view', 'cross-section',
        'exploded view', 'layout', 'blueprint', 'graph'
    }

    def __init__(self, clip_embedder=None, name: str = "Image"):
        super().__init__(name=name, expert_type=ExpertType.IMAGE)
        self.clip_embedder = clip_embedder

    def score(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 20
    ) -> ExpertResult:
        """多模态评分"""
        import time
        start = time.time()

        raw_scores = []
        for doc in candidates:
            score = 0.0
            content = doc.get('content', '').lower()

            # 基于视觉关键词的启发式分数
            visual_count = sum(1 for kw in self.VISUAL_KEYWORDS if kw in content)
            score += visual_count * 0.3

            # 如果文档有图像路径或是图像类型
            if doc.get('image_path') or doc.get('modality') == 'image':
                score += 2.0

            # 如果有图像摘要
            if doc.get('summary') or doc.get('caption'):
                summary_text = doc.get('summary', '') or doc.get('caption', '')
                # 简单词重叠
                q_tokens = set(query.lower().split())
                s_tokens = set(summary_text.lower().split())
                overlap = len(q_tokens & s_tokens)
                score += overlap * 0.5

            # CLIP 嵌入相似度 (如果可用)
            if self.clip_embedder is not None and doc.get('image_embedding') is not None:
                try:
                    q_emb = np.array(
                        self.clip_embedder.embed_text(query), dtype=np.float32
                    )
                    d_emb = np.array(doc['image_embedding'], dtype=np.float32)
                    sim = np.dot(q_emb, d_emb) / (
                        np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10
                    )
                    score += float(sim) * 3.0
                except Exception:
                    pass

            raw_scores.append(score)

        normed = self._normalize_scores(raw_scores)

        scored_docs = []
        for doc, ns in zip(candidates, normed):
            scored_doc = doc.copy()
            scored_doc['expert_score'] = ns
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['expert_score'], reverse=True)
        latency = (time.time() - start) * 1000

        return ExpertResult(
            expert_name=self.name,
            expert_type=self.expert_type,
            scored_docs=scored_docs[:top_k],
            latency_ms=latency
        )


def create_default_experts(
    embedder=None,
    clip_embedder=None,
    entity_kb=None
) -> List[BaseExpert]:
    """
    创建默认的四个专家

    Args:
        embedder: 文本嵌入器
        clip_embedder: CLIP 嵌入器
        entity_kb: 实体知识库

    Returns:
        4个专家的列表 [BM25, Dense, Entity, Image]
    """
    return [
        BM25Expert(),
        DenseExpert(embedder=embedder),
        EntityExpert(entity_kb=entity_kb),
        ImageExpert(clip_embedder=clip_embedder)
    ]
