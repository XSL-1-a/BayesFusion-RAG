"""
N-Expert 最优权重理论
将 2-Expert α 理论推广到 N 个检索专家的一般情形

理论框架:
1. 信息论最优权重:
   w*_i = (ρ_i / σ²_i) / Σ_j (ρ_j / σ²_j)
   其中 ρ_i = corr(expert_i_scores, relevance), σ²_i = var(expert_i_scores)

2. 贝叶斯后验权重:
   w*_i = LR_i / Σ_j LR_j
   其中 LR_i = E[S_i|R=1] - E[S_i|R=0] (似然比)

3. 经验验证:
   通过组合优化搜索验证理论公式的近优性

与 alpha_theory.py 的关系:
- alpha_theory.py: 2-Expert 特殊情况 (BM25 + Dense)
- n_expert_theory.py: N-Expert 一般情况 (BM25 + Dense + Entity + Image)
- 当 N=2 时，本模块退化为 alpha_theory.py 的结果

作者: RAG Research Team
日期: 2026-01-28
"""

import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import warnings
import logging

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


class NExpertTheoryAnalyzer:
    """
    N-Expert 最优融合权重理论分析器

    给定 N 个专家的评分和真实相关性标签，
    计算信息论最优权重、贝叶斯权重和经验最优权重，
    并验证三者的一致性
    """

    def __init__(self, expert_names: Optional[List[str]] = None):
        """
        Args:
            expert_names: 专家名称列表
        """
        self.expert_names = expert_names or ['BM25', 'Dense', 'Entity', 'Image']
        self.n_experts = len(self.expert_names)

    def compute_optimal_weights_information_theory(
        self,
        expert_scores: Dict[str, np.ndarray],
        relevance_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        信息论框架下的 N-Expert 最优权重

        公式: w*_i = (ρ_i / σ²_i) / Σ_j (ρ_j / σ²_j)

        直觉:
        - ρ_i 高 → 专家 i 与真实相关性相关度高 → 权重大
        - σ²_i 高 → 专家 i 噪声大 → 权重小
        - 这是 "信噪比" 准则: 权重正比于 信号强度/噪声水平

        Args:
            expert_scores: {expert_name: np.ndarray(n_samples,)}
            relevance_labels: np.ndarray(n_samples,)

        Returns:
            包含最优权重和参数的字典
        """
        expert_params = {}
        raw_weights = {}

        for name in self.expert_names:
            if name not in expert_scores:
                continue

            scores = self._normalize_scores(expert_scores[name])
            rho, p_value = stats.pearsonr(scores, relevance_labels)
            var = np.var(scores, ddof=1)

            expert_params[name] = {
                'rho': float(rho),
                'p_value': float(p_value),
                'var': float(var),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }

            # w*_i ∝ |ρ_i| / σ²_i
            var_safe = max(var, 1e-6)
            raw_weights[name] = abs(rho) / var_safe

        # 归一化到单纯形
        total = sum(raw_weights.values())
        if total > 1e-10:
            optimal_weights = {k: v / total for k, v in raw_weights.items()}
        else:
            n = len(raw_weights)
            optimal_weights = {k: 1.0 / n for k in raw_weights}

        return {
            'optimal_weights': optimal_weights,
            'expert_params': expert_params,
            'framework': 'information_theory',
            'formula': 'w*_i = (|ρ_i| / σ²_i) / Σ_j (|ρ_j| / σ²_j)',
            'n_samples': len(relevance_labels),
            'n_experts': len(optimal_weights)
        }

    def compute_optimal_weights_bayesian(
        self,
        expert_scores: Dict[str, np.ndarray],
        relevance_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        贝叶斯后验框架下的 N-Expert 最优权重

        公式: w*_i = LR_i / Σ_j LR_j
        其中 LR_i = E[S_i|R=1] - E[S_i|R=0] (判别力)

        直觉: 权重正比于专家区分相关/不相关文档的能力

        Args:
            expert_scores: {expert_name: np.ndarray(n_samples,)}
            relevance_labels: np.ndarray(n_samples,)
        """
        rel_mask = relevance_labels == 1
        irrel_mask = relevance_labels == 0
        n_rel = np.sum(rel_mask)
        n_irrel = np.sum(irrel_mask)

        if n_rel < 5 or n_irrel < 5:
            return {
                'optimal_weights': {
                    name: 1.0 / self.n_experts for name in self.expert_names
                },
                'error': 'Insufficient samples for Bayesian estimation',
                'framework': 'bayesian'
            }

        expert_params = {}
        likelihood_ratios = {}

        for name in self.expert_names:
            if name not in expert_scores:
                continue

            scores = self._normalize_scores(expert_scores[name])

            mean_rel = np.mean(scores[rel_mask])
            mean_irrel = np.mean(scores[irrel_mask])
            std_rel = np.std(scores[rel_mask])
            std_irrel = np.std(scores[irrel_mask])

            # 判别力: 相关文档平均分 - 不相关文档平均分
            lr = max(mean_rel - mean_irrel, 0.01)

            # Cohen's d 效应量
            pooled_std = np.sqrt(
                ((n_rel - 1) * std_rel**2 + (n_irrel - 1) * std_irrel**2) /
                (n_rel + n_irrel - 2)
            )
            cohens_d = (mean_rel - mean_irrel) / (pooled_std + 1e-10)

            expert_params[name] = {
                'mean_relevant': float(mean_rel),
                'mean_irrelevant': float(mean_irrel),
                'std_relevant': float(std_rel),
                'std_irrelevant': float(std_irrel),
                'likelihood_ratio': float(lr),
                'cohens_d': float(cohens_d)
            }
            likelihood_ratios[name] = lr

        # 归一化
        total = sum(likelihood_ratios.values())
        optimal_weights = {k: v / total for k, v in likelihood_ratios.items()}

        return {
            'optimal_weights': optimal_weights,
            'expert_params': expert_params,
            'framework': 'bayesian',
            'formula': 'w*_i = LR_i / Σ_j LR_j',
            'n_relevant': int(n_rel),
            'n_irrelevant': int(n_irrel)
        }

    def compute_optimal_weights_empirical(
        self,
        expert_scores: Dict[str, np.ndarray],
        relevance_labels: np.ndarray,
        metric: str = 'ndcg',
        n_random: int = 10000
    ) -> Dict[str, Any]:
        """
        经验最优权重 (随机搜索 + 局部优化)

        通过在单纯形上搜索来找到使评估指标最优的权重组合

        Args:
            expert_scores: {expert_name: np.ndarray(n_samples,)}
            relevance_labels: np.ndarray(n_samples,)
            metric: 评估指标 ('ndcg', 'map', 'recall')
            n_random: 随机搜索的采样数量
        """
        active_names = [n for n in self.expert_names if n in expert_scores]
        n = len(active_names)

        if n == 0:
            return {'error': 'No expert scores provided'}

        # 归一化分数
        normed_scores = {
            name: self._normalize_scores(expert_scores[name])
            for name in active_names
        }

        score_matrix = np.stack(
            [normed_scores[name] for name in active_names], axis=0
        )  # (n_experts, n_samples)

        # 评估函数
        def evaluate_weights(weights):
            fused = weights @ score_matrix  # (n_samples,)
            if metric == 'ndcg':
                return self._compute_ndcg(fused, relevance_labels, k=5)
            elif metric == 'map':
                return self._compute_map(fused, relevance_labels)
            else:
                return self._compute_recall_at_k(fused, relevance_labels, k=5)

        # 随机搜索
        best_score = -1
        best_weights = np.ones(n) / n

        # Dirichlet 采样生成单纯形上的随机点
        for _ in range(n_random):
            w = np.random.dirichlet(np.ones(n))
            score = evaluate_weights(w)
            if score > best_score:
                best_score = score
                best_weights = w

        # 局部优化 (SLSQP)
        def neg_metric(w):
            return -evaluate_weights(w)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.01, 0.99)] * n

        try:
            result = optimize.minimize(
                neg_metric, best_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            if result.success and -result.fun > best_score:
                best_weights = result.x
                best_score = -result.fun
        except Exception:
            pass

        # 归一化
        best_weights = best_weights / best_weights.sum()

        optimal_weights = {
            active_names[i]: float(best_weights[i]) for i in range(n)
        }

        # 计算各种权重下的性能 (用于绘图)
        weight_performance = self._compute_weight_sweep(
            score_matrix, relevance_labels, active_names, metric
        )

        return {
            'optimal_weights': optimal_weights,
            'best_metric_value': float(best_score),
            'metric_name': metric,
            'framework': 'empirical',
            'weight_performance': weight_performance,
            'n_random_samples': n_random
        }

    def validate_theory_vs_empirical(
        self,
        expert_scores: Dict[str, np.ndarray],
        relevance_labels: np.ndarray,
        metric: str = 'ndcg'
    ) -> Dict[str, Any]:
        """
        验证理论权重与经验最优权重的一致性

        计算三种框架的最优权重并比较
        """
        info_result = self.compute_optimal_weights_information_theory(
            expert_scores, relevance_labels
        )
        bayes_result = self.compute_optimal_weights_bayesian(
            expert_scores, relevance_labels
        )
        empirical_result = self.compute_optimal_weights_empirical(
            expert_scores, relevance_labels, metric=metric
        )

        w_info = info_result['optimal_weights']
        w_bayes = bayes_result['optimal_weights']
        w_emp = empirical_result['optimal_weights']

        # 计算各框架权重下的实际指标
        active_names = [n for n in self.expert_names if n in expert_scores]
        normed_scores = {
            name: self._normalize_scores(expert_scores[name])
            for name in active_names
        }
        score_matrix = np.stack(
            [normed_scores[name] for name in active_names], axis=0
        )

        def eval_weights(weights_dict):
            w = np.array([weights_dict.get(n, 0) for n in active_names])
            w = w / (w.sum() + 1e-10)
            fused = w @ score_matrix
            if metric == 'ndcg':
                return self._compute_ndcg(fused, relevance_labels, k=5)
            return self._compute_recall_at_k(fused, relevance_labels, k=5)

        metric_info = eval_weights(w_info)
        metric_bayes = eval_weights(w_bayes)
        metric_emp = empirical_result['best_metric_value']
        metric_uniform = eval_weights(
            {n: 1.0 / len(active_names) for n in active_names}
        )

        # 权重距离 (L1 距离)
        def weight_distance(w1, w2):
            names = set(w1.keys()) | set(w2.keys())
            return sum(abs(w1.get(n, 0) - w2.get(n, 0)) for n in names)

        return {
            'weights': {
                'information_theory': w_info,
                'bayesian': w_bayes,
                'empirical': w_emp,
                'uniform': {n: 1.0 / len(active_names) for n in active_names}
            },
            'metrics': {
                'information_theory': float(metric_info),
                'bayesian': float(metric_bayes),
                'empirical': float(metric_emp),
                'uniform': float(metric_uniform)
            },
            'distances': {
                'info_vs_empirical': float(weight_distance(w_info, w_emp)),
                'bayes_vs_empirical': float(weight_distance(w_bayes, w_emp)),
                'info_vs_bayes': float(weight_distance(w_info, w_bayes))
            },
            'improvement_over_uniform': {
                'information_theory': float(
                    (metric_info - metric_uniform) / (metric_uniform + 1e-10) * 100
                ),
                'bayesian': float(
                    (metric_bayes - metric_uniform) / (metric_uniform + 1e-10) * 100
                ),
                'empirical': float(
                    (metric_emp - metric_uniform) / (metric_uniform + 1e-10) * 100
                )
            },
            'metric_name': metric,
            'expert_params': {
                'info': info_result.get('expert_params', {}),
                'bayes': bayes_result.get('expert_params', {})
            }
        }

    def cross_domain_analysis(
        self,
        domain_data: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]],
        metric: str = 'ndcg'
    ) -> Dict[str, Any]:
        """
        跨域分析: 验证 N-Expert 理论在不同领域的一致性

        Args:
            domain_data: {domain_name: (expert_scores_dict, relevance_labels)}
            metric: 评估指标
        """
        domain_results = {}
        all_info_weights = {name: [] for name in self.expert_names}
        all_emp_weights = {name: [] for name in self.expert_names}

        for domain_name, (expert_scores, rel_labels) in domain_data.items():
            validation = self.validate_theory_vs_empirical(
                expert_scores, rel_labels, metric=metric
            )
            domain_results[domain_name] = validation

            for name in self.expert_names:
                w_info = validation['weights']['information_theory'].get(name, 0)
                w_emp = validation['weights']['empirical'].get(name, 0)
                all_info_weights[name].append(w_info)
                all_emp_weights[name].append(w_emp)

        # 跨域统计
        cross_stats = {}
        for name in self.expert_names:
            info_w = np.array(all_info_weights[name])
            emp_w = np.array(all_emp_weights[name])

            if len(info_w) > 1:
                corr, p_val = stats.pearsonr(info_w, emp_w)
            else:
                corr, p_val = 0.0, 1.0

            cross_stats[name] = {
                'mean_info_weight': float(np.mean(info_w)),
                'std_info_weight': float(np.std(info_w)),
                'mean_emp_weight': float(np.mean(emp_w)),
                'std_emp_weight': float(np.std(emp_w)),
                'cross_domain_correlation': float(corr),
                'p_value': float(p_val)
            }

        # 全局一致性: 理论权重vs经验权重的平均距离
        all_distances = []
        for domain_name, result in domain_results.items():
            all_distances.append(result['distances']['info_vs_empirical'])

        return {
            'domain_results': domain_results,
            'cross_domain_expert_stats': cross_stats,
            'global_stats': {
                'mean_theory_empirical_distance': float(np.mean(all_distances)),
                'std_theory_empirical_distance': float(np.std(all_distances)),
                'n_domains': len(domain_data)
            }
        }

    def connection_to_2expert(
        self,
        expert_scores: Dict[str, np.ndarray],
        relevance_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        展示 N=2 时与 alpha_theory.py 的等价性

        当只有 BM25 和 Dense 两个专家时:
        - N-Expert: w_BM25 = (ρ_BM25/σ²_BM25) / (ρ_BM25/σ²_BM25 + ρ_Dense/σ²_Dense)
        - α-theory: α = (σ²_Dense × ρ_BM25) / (σ²_BM25 × ρ_Dense + σ²_Dense × ρ_BM25)
        - 两者等价!
        """
        # N-Expert 结果
        n_expert_result = self.compute_optimal_weights_information_theory(
            expert_scores, relevance_labels
        )

        # 手动计算 alpha_theory 结果 (2-expert special case)
        bm25_key = None
        dense_key = None
        for name in expert_scores:
            if 'bm25' in name.lower():
                bm25_key = name
            elif 'dense' in name.lower():
                dense_key = name

        if bm25_key is None or dense_key is None:
            return {
                'error': 'Need BM25 and Dense experts for 2-expert comparison',
                'n_expert_weights': n_expert_result['optimal_weights']
            }

        bm25_normed = self._normalize_scores(expert_scores[bm25_key])
        dense_normed = self._normalize_scores(expert_scores[dense_key])

        rho_bm25, _ = stats.pearsonr(bm25_normed, relevance_labels)
        rho_dense, _ = stats.pearsonr(dense_normed, relevance_labels)
        var_bm25 = np.var(bm25_normed, ddof=1)
        var_dense = np.var(dense_normed, ddof=1)

        # alpha_theory 公式
        numerator = var_dense * abs(rho_bm25)
        denominator = var_bm25 * abs(rho_dense) + var_dense * abs(rho_bm25)
        if denominator > 1e-10:
            alpha_2expert = numerator / denominator
        else:
            alpha_2expert = 0.5

        # N-expert 公式中 BM25 的权重
        w_bm25_n_expert = n_expert_result['optimal_weights'].get(bm25_key, 0.5)

        return {
            'alpha_2expert': float(np.clip(alpha_2expert, 0.05, 0.95)),
            'w_bm25_n_expert': float(w_bm25_n_expert),
            'absolute_difference': float(abs(alpha_2expert - w_bm25_n_expert)),
            'are_equivalent': abs(alpha_2expert - w_bm25_n_expert) < 0.01,
            'n_expert_weights': n_expert_result['optimal_weights'],
            'parameters': {
                'rho_bm25': float(rho_bm25),
                'rho_dense': float(rho_dense),
                'var_bm25': float(var_bm25),
                'var_dense': float(var_dense)
            }
        }

    def export_latex_formula(self) -> str:
        """导出 N-Expert 理论的 LaTeX 公式"""
        return r"""
% N-Expert Optimal Weight Theory
% Generalization of 2-Expert α formula to N experts

% Information-Theoretic Framework
\begin{theorem}[N-Expert Optimal Weights]
Given $N$ retrieval experts with scores $\{S_i\}_{i=1}^{N}$ and ground truth relevance $R$,
the information-theoretically optimal fusion weights are:
\begin{equation}
w_i^* = \frac{|\rho_i| / \sigma_i^2}{\sum_{j=1}^{N} |\rho_j| / \sigma_j^2}
\label{eq:n_expert_optimal}
\end{equation}
where $\rho_i = \text{Corr}(S_i, R)$ is the Pearson correlation between expert $i$'s scores
and relevance, and $\sigma_i^2 = \text{Var}(S_i)$ is the variance of expert $i$'s scores.
\end{theorem}

% Connection to 2-Expert Case
\begin{corollary}[Equivalence to $\alpha$ Formula]
When $N=2$ with experts BM25 and Dense, Equation~\ref{eq:n_expert_optimal} reduces to:
\begin{equation}
\alpha^* = w_{\text{BM25}}^* = \frac{\sigma_{\text{Dense}}^2 \cdot |\rho_{\text{BM25}}|}{\sigma_{\text{BM25}}^2 \cdot |\rho_{\text{Dense}}| + \sigma_{\text{Dense}}^2 \cdot |\rho_{\text{BM25}}|}
\end{equation}
which is exactly the formula derived in Section~\ref{sec:alpha_theory}.
\end{corollary}

% Bayesian Framework
\begin{proposition}[Bayesian N-Expert Weights]
Under a Bayesian framework with likelihood ratio $\text{LR}_i = \mathbb{E}[S_i|R=1] - \mathbb{E}[S_i|R=0]$:
\begin{equation}
w_i^* = \frac{\text{LR}_i}{\sum_{j=1}^{N} \text{LR}_j}
\end{equation}
\end{proposition}

% Fusion Score
\begin{equation}
S_{\text{MoRE}} = \sum_{i=1}^{N} w_i \cdot S_i, \quad \text{s.t. } \sum_{i=1}^{N} w_i = 1, \; w_i \geq 0
\end{equation}
"""

    def _compute_weight_sweep(
        self,
        score_matrix: np.ndarray,
        relevance_labels: np.ndarray,
        expert_names: List[str],
        metric: str,
        n_sweep: int = 50
    ) -> List[Dict]:
        """在单纯形上进行权重扫描"""
        n_experts = score_matrix.shape[0]
        results = []

        if n_experts == 2:
            for alpha in np.linspace(0.0, 1.0, n_sweep):
                w = np.array([alpha, 1 - alpha])
                fused = w @ score_matrix
                if metric == 'ndcg':
                    val = self._compute_ndcg(fused, relevance_labels, k=5)
                else:
                    val = self._compute_recall_at_k(fused, relevance_labels, k=5)
                results.append({
                    'weights': {expert_names[i]: float(w[i]) for i in range(n_experts)},
                    'metric_value': float(val)
                })
        else:
            # 对N>2，使用 Dirichlet 采样
            for _ in range(n_sweep * 10):
                w = np.random.dirichlet(np.ones(n_experts))
                fused = w @ score_matrix
                if metric == 'ndcg':
                    val = self._compute_ndcg(fused, relevance_labels, k=5)
                else:
                    val = self._compute_recall_at_k(fused, relevance_labels, k=5)
                results.append({
                    'weights': {expert_names[i]: float(w[i]) for i in range(n_experts)},
                    'metric_value': float(val)
                })

        results.sort(key=lambda x: x['metric_value'], reverse=True)
        return results[:n_sweep]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]"""
        scores = np.array(scores, dtype=float)
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val > 1e-10:
            return (scores - min_val) / (max_val - min_val)
        return np.full_like(scores, 0.5)

    def _compute_ndcg(
        self, scores: np.ndarray, relevance: np.ndarray, k: int = 5
    ) -> float:
        """计算 NDCG@k"""
        sorted_idx = np.argsort(scores)[::-1][:k]
        sorted_rel = relevance[sorted_idx]
        dcg = np.sum(sorted_rel / np.log2(np.arange(2, len(sorted_rel) + 2)))
        ideal_rel = np.sort(relevance)[::-1][:k]
        idcg = np.sum(ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2)))
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(self, scores: np.ndarray, relevance: np.ndarray) -> float:
        """计算 MAP"""
        sorted_idx = np.argsort(scores)[::-1]
        sorted_rel = relevance[sorted_idx]
        n_rel = np.sum(relevance)
        if n_rel == 0:
            return 0.0
        precisions = []
        n_hits = 0
        for i, rel in enumerate(sorted_rel):
            if rel == 1:
                n_hits += 1
                precisions.append(n_hits / (i + 1))
        return np.mean(precisions) if precisions else 0.0

    def _compute_recall_at_k(
        self, scores: np.ndarray, relevance: np.ndarray, k: int = 5
    ) -> float:
        """计算 Recall@k"""
        sorted_idx = np.argsort(scores)[::-1][:k]
        retrieved_rel = relevance[sorted_idx]
        n_total_rel = np.sum(relevance)
        if n_total_rel == 0:
            return 0.0
        return np.sum(retrieved_rel) / n_total_rel
