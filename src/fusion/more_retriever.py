"""
MoRE (Mixture-of-Retrieval-Experts) 主检索器
统一框架：将 BM25/Dense/Entity/Image 视为检索专家
通过学习式门控网络实现查询自适应的专家权重分配

核心流程:
1. 特征提取: 从查询中提取门控特征
2. 门控路由: 门控网络预测 N 维专家权重
3. 专家检索: 各专家独立对候选文档评分
4. 加权融合: 按门控权重融合专家分数
5. 结果排序: 按融合分数排序返回 top-k

创新点:
- 统一的专家化框架 (vs. 硬编码权重)
- 负载均衡 + 专家化正则 (避免专家退化)
- N-Expert 理论提供可解释的最优权重上界
- 与 ECG 门控、α理论形成统一理论体系

作者: RAG Research Team
日期: 2026-01-28
"""

import time
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .more_features import MoREFeatureExtractor, MoREFeatures
from .gating_network import MoREGatingNet, GatingOutput
from .retrieval_experts import (
    BaseExpert, ExpertResult, ExpertType,
    BM25Expert, DenseExpert, EntityExpert, ImageExpert,
    create_default_experts
)

logger = logging.getLogger(__name__)


@dataclass
class MoREResult:
    """MoRE 检索结果"""
    # 最终融合结果
    results: List[Dict[str, Any]]
    # 门控信息
    expert_weights: Dict[str, float]      # {expert_name: weight}
    top_expert: str                       # 权重最高的专家
    # 各专家独立结果
    expert_results: Dict[str, ExpertResult]
    # 统计信息
    feature_dim: int
    total_latency_ms: float
    gating_latency_ms: float
    expert_latencies: Dict[str, float]
    # 元信息
    query: str
    n_candidates: int
    n_returned: int


class MoRERetriever:
    """
    MoRE (Mixture-of-Retrieval-Experts) 检索器

    将检索视为多专家混合问题：
    - 每种检索方法是一个"专家"
    - 门控网络根据查询特征动态分配专家权重
    - 最终结果是各专家评分的加权和

    支持两种模式:
    1. 学习模式 (learned): 使用训练好的门控网络
    2. 理论模式 (theory): 使用 N-Expert 理论公式计算权重
    """

    def __init__(
        self,
        experts: Optional[List[BaseExpert]] = None,
        gating_net: Optional[MoREGatingNet] = None,
        feature_extractor: Optional[MoREFeatureExtractor] = None,
        mode: str = "learned",
        device: str = "cpu",
        n_experts: int = 4,
        query_dim: int = 3072
    ):
        """
        Args:
            experts: 检索专家列表
            gating_net: 门控网络 (learned 模式必需)
            feature_extractor: 特征提取器
            mode: "learned" 使用门控网络, "theory" 使用理论权重, "uniform" 均匀权重
            device: 计算设备
            n_experts: 专家数量
            query_dim: 查询嵌入维度
        """
        self.experts = experts or create_default_experts()
        self.n_experts = len(self.experts)
        self.mode = mode
        self.device = device

        self.feature_extractor = feature_extractor or MoREFeatureExtractor(
            query_dim=query_dim
        )

        if gating_net is not None:
            self.gating_net = gating_net
        elif mode == "learned":
            self.gating_net = MoREGatingNet(
                feature_dim=self.feature_extractor.feature_dim,
                n_experts=self.n_experts
            )
        else:
            self.gating_net = None

        # 理论模式的缓存参数 (从训练数据估计)
        self._theory_params: Optional[Dict] = None

        logger.info(
            f"MoRERetriever initialized: mode={mode}, "
            f"experts={[e.name for e in self.experts]}"
        )

    def retrieve(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> MoREResult:
        """
        MoRE 检索主流程

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回的结果数量

        Returns:
            MoREResult 包含融合结果和详细统计
        """
        total_start = time.time()

        # Step 1: 特征提取
        features = self.feature_extractor.extract_features(query, candidates)

        # Step 2: 获取专家权重
        gating_start = time.time()
        expert_weights = self._get_expert_weights(features)
        gating_latency = (time.time() - gating_start) * 1000

        # Step 3: 各专家评分
        expert_results = {}
        expert_latencies = {}
        expert_score_matrix = {}  # {expert_name: {doc_id: score}}

        for expert in self.experts:
            result = expert.score(query, candidates, top_k=len(candidates))
            expert_results[expert.name] = result
            expert_latencies[expert.name] = result.latency_ms

            score_map = {}
            for doc in result.scored_docs:
                doc_id = doc.get('doc_id', doc.get('content', '')[:50])
                score_map[doc_id] = doc.get('expert_score', 0.0)
            expert_score_matrix[expert.name] = score_map

        # Step 4: 加权融合
        fused_results = self._fuse_scores(
            candidates, expert_weights, expert_score_matrix
        )

        # Step 5: 排序并返回 top-k
        fused_results.sort(key=lambda x: x.get('fused_score', 0), reverse=True)
        top_results = fused_results[:top_k]

        total_latency = (time.time() - total_start) * 1000

        # 找到权重最高的专家
        top_expert = max(expert_weights, key=expert_weights.get)

        return MoREResult(
            results=top_results,
            expert_weights=expert_weights,
            top_expert=top_expert,
            expert_results=expert_results,
            feature_dim=self.feature_extractor.feature_dim,
            total_latency_ms=total_latency,
            gating_latency_ms=gating_latency,
            expert_latencies=expert_latencies,
            query=query,
            n_candidates=len(candidates),
            n_returned=len(top_results)
        )

    def _get_expert_weights(self, features: MoREFeatures) -> Dict[str, float]:
        """
        获取专家权重

        根据 mode 选择不同策略:
        - learned: 门控网络推理
        - theory: N-Expert 理论公式
        - uniform: 均匀分配
        """
        if self.mode == "learned" and self.gating_net is not None:
            return self._get_learned_weights(features)
        elif self.mode == "theory":
            return self._get_theory_weights(features)
        else:
            # uniform
            w = 1.0 / self.n_experts
            return {e.name: w for e in self.experts}

    def _get_learned_weights(self, features: MoREFeatures) -> Dict[str, float]:
        """使用门控网络获取权重"""
        import torch

        self.gating_net.eval()
        feature_tensor = self.feature_extractor.to_tensor(features)
        feature_tensor = feature_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            gating_out = self.gating_net(feature_tensor)
            weights = gating_out.expert_weights[0].cpu().numpy()

        return {
            self.experts[i].name: float(weights[i])
            for i in range(self.n_experts)
        }

    def _get_theory_weights(self, features: MoREFeatures) -> Dict[str, float]:
        """
        使用 N-Expert 理论公式获取权重

        w*_i = (ρ_i / σ²_i) / Σ_j (ρ_j / σ²_j)

        需要预先通过 set_theory_params 设置各专家的 ρ 和 σ²
        """
        if self._theory_params is None:
            # 没有理论参数时回退到均匀分配
            w = 1.0 / self.n_experts
            return {e.name: w for e in self.experts}

        weights = {}
        total = 0.0

        for expert in self.experts:
            params = self._theory_params.get(expert.name, {})
            rho = abs(params.get('rho', 0.5))
            var = params.get('var', 1.0)
            var = max(var, 1e-6)  # 防止除零

            w_raw = rho / var
            weights[expert.name] = w_raw
            total += w_raw

        # 归一化
        if total > 1e-10:
            for name in weights:
                weights[name] /= total
        else:
            w = 1.0 / self.n_experts
            weights = {e.name: w for e in self.experts}

        return weights

    def set_theory_params(self, params: Dict[str, Dict[str, float]]):
        """
        设置 N-Expert 理论参数

        Args:
            params: {expert_name: {'rho': float, 'var': float}}
        """
        self._theory_params = params

    def _fuse_scores(
        self,
        candidates: List[Dict],
        expert_weights: Dict[str, float],
        expert_score_matrix: Dict[str, Dict[str, float]]
    ) -> List[Dict]:
        """
        加权融合各专家分数

        fused_score = Σ w_i × expert_score_i
        """
        fused = []
        for doc in candidates:
            doc_id = doc.get('doc_id', doc.get('content', '')[:50])

            fused_score = 0.0
            expert_scores_detail = {}

            for expert_name, weight in expert_weights.items():
                score_map = expert_score_matrix.get(expert_name, {})
                expert_score = score_map.get(doc_id, 0.0)
                fused_score += weight * expert_score
                expert_scores_detail[expert_name] = expert_score

            result = doc.copy()
            result['fused_score'] = fused_score
            result['expert_scores'] = expert_scores_detail
            result['score'] = fused_score  # 兼容下游
            fused.append(result)

        return fused

    def load_gating_network(self, checkpoint_path: str):
        """加载训练好的门控网络"""
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('model_config', {})

        self.gating_net = MoREGatingNet(
            feature_dim=config.get('feature_dim', self.feature_extractor.feature_dim),
            n_experts=config.get('n_experts', self.n_experts),
            hidden_dim=config.get('hidden_dim', 256),
            temperature=config.get('temperature', 1.0),
            top_k=config.get('top_k', 0)
        ).to(self.device)

        self.gating_net.load_state_dict(checkpoint['model_state_dict'])
        self.gating_net.eval()
        self.mode = "learned"
        logger.info(f"Loaded gating network from {checkpoint_path}")

    def get_statistics(self, result: MoREResult) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            'query': result.query,
            'mode': self.mode,
            'n_experts': self.n_experts,
            'expert_names': [e.name for e in self.experts],
            'expert_weights': result.expert_weights,
            'top_expert': result.top_expert,
            'n_candidates': result.n_candidates,
            'n_returned': result.n_returned,
            'total_latency_ms': result.total_latency_ms,
            'gating_latency_ms': result.gating_latency_ms,
            'expert_latencies': result.expert_latencies,
            'feature_dim': result.feature_dim,
            'score_range': {
                'min': min(r['fused_score'] for r in result.results) if result.results else 0,
                'max': max(r['fused_score'] for r in result.results) if result.results else 0
            }
        }

    def analyze_expert_routing(
        self,
        queries: List[str],
        candidates_list: List[List[Dict]]
    ) -> Dict[str, Any]:
        """
        分析多个查询的专家路由模式

        Args:
            queries: 查询列表
            candidates_list: 每个查询对应的候选文档列表

        Returns:
            路由分析统计
        """
        all_weights = []
        top_expert_counts = {e.name: 0 for e in self.experts}
        query_types = []

        for query, candidates in zip(queries, candidates_list):
            result = self.retrieve(query, candidates, top_k=5)
            all_weights.append(list(result.expert_weights.values()))
            top_expert_counts[result.top_expert] += 1
            query_types.append(
                self.feature_extractor.classify_query_type(query)
            )

        weights_array = np.array(all_weights)
        n_queries = len(queries)

        return {
            'n_queries': n_queries,
            'avg_weights': {
                self.experts[i].name: float(weights_array[:, i].mean())
                for i in range(self.n_experts)
            },
            'std_weights': {
                self.experts[i].name: float(weights_array[:, i].std())
                for i in range(self.n_experts)
            },
            'top_expert_distribution': {
                k: v / n_queries for k, v in top_expert_counts.items()
            },
            'weight_entropy': float(
                -np.sum(
                    weights_array.mean(axis=0) *
                    np.log(weights_array.mean(axis=0) + 1e-10)
                )
            ),
            'query_type_distribution': {
                qt: query_types.count(qt) / n_queries
                for qt in set(query_types)
            }
        }
