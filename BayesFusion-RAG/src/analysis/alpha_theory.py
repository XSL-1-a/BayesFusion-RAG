"""
HMR α权重理论分析模块
提供信息论和贝叶斯框架下的理论解释

作者: RAG Research Team
日期: 2026-01-28
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class AlphaTheoryAnalyzer:
    """
    HMR 融合权重 α 的理论分析器

    理论框架:
    1. 信息论框架: 基于相关系数和方差的最优权重
       α* = σ²_Dense × ρ_BM25 / (σ²_BM25 × ρ_Dense + σ²_Dense × ρ_BM25)

    2. 贝叶斯框架: 基于后验概率的可信度分配
       P(R=1|S_BM25, S_Dense) ∝ P(S_BM25|R=1)^α × P(S_Dense|R=1)^(1-α)

    3. 经验验证: 通过网格搜索找到最优 α
    """

    def __init__(self):
        self.results = {}
        self.alpha_grid = np.linspace(0.05, 0.95, 19)

    def compute_optimal_alpha_information_theory(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        relevance_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        基于信息论框架计算最优 α

        理论公式:
        α* = (σ²_Dense × ρ_BM25) / (σ²_BM25 × ρ_Dense + σ²_Dense × ρ_BM25)
        """
        bm25_normed = self._normalize_scores(bm25_scores)
        dense_normed = self._normalize_scores(dense_scores)

        rho_bm25, p_bm25 = stats.pearsonr(bm25_normed, relevance_labels)
        rho_dense, p_dense = stats.pearsonr(dense_normed, relevance_labels)

        var_bm25 = np.var(bm25_normed, ddof=1)
        var_dense = np.var(dense_normed, ddof=1)

        numerator = var_dense * abs(rho_bm25)
        denominator = var_bm25 * abs(rho_dense) + var_dense * abs(rho_bm25)

        if denominator > 1e-10:
            alpha_theory = numerator / denominator
        else:
            alpha_theory = 0.5

        alpha_theory = np.clip(alpha_theory, 0.05, 0.95)

        return {
            'alpha_theory': float(alpha_theory),
            'rho_bm25': float(rho_bm25),
            'rho_dense': float(rho_dense),
            'p_value_bm25': float(p_bm25),
            'p_value_dense': float(p_dense),
            'var_bm25': float(var_bm25),
            'var_dense': float(var_dense),
            'n_samples': len(bm25_scores)
        }

    def compute_optimal_alpha_bayesian(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        relevance_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        基于贝叶斯后验框架计算最优 α
        """
        bm25_normed = self._normalize_scores(bm25_scores)
        dense_normed = self._normalize_scores(dense_scores)

        rel_mask = relevance_labels == 1
        irrel_mask = relevance_labels == 0

        n_rel = np.sum(rel_mask)
        n_irrel = np.sum(irrel_mask)

        if n_rel < 5 or n_irrel < 5:
            return {
                'alpha_bayesian': 0.5,
                'likelihood_ratio_bm25': 1.0,
                'likelihood_ratio_dense': 1.0,
                'error': 'Insufficient samples'
            }

        mean_bm25_rel = np.mean(bm25_normed[rel_mask])
        mean_bm25_irrel = np.mean(bm25_normed[irrel_mask])
        mean_dense_rel = np.mean(dense_normed[rel_mask])
        mean_dense_irrel = np.mean(dense_normed[irrel_mask])

        lr_bm25 = max(mean_bm25_rel - mean_bm25_irrel, 0.01)
        lr_dense = max(mean_dense_rel - mean_dense_irrel, 0.01)

        alpha_bayesian = lr_dense / (lr_bm25 + lr_dense)
        alpha_bayesian = np.clip(alpha_bayesian, 0.05, 0.95)

        return {
            'alpha_bayesian': float(alpha_bayesian),
            'likelihood_ratio_bm25': float(lr_bm25),
            'likelihood_ratio_dense': float(lr_dense),
            'n_relevant': int(n_rel),
            'n_irrelevant': int(n_irrel)
        }

    def compute_optimal_alpha_empirical(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        relevance_labels: np.ndarray,
        metric: str = 'ndcg'
    ) -> Dict[str, Any]:
        """通过网格搜索计算经验最优 α"""
        bm25_normed = self._normalize_scores(bm25_scores)
        dense_normed = self._normalize_scores(dense_scores)

        alpha_values = []
        metric_values = []

        for alpha in self.alpha_grid:
            fused_scores = (1 - alpha) * dense_normed + alpha * bm25_normed

            if metric == 'ndcg':
                score = self._compute_ndcg(fused_scores, relevance_labels, k=5)
            elif metric == 'map':
                score = self._compute_map(fused_scores, relevance_labels)
            else:
                score = self._compute_recall_at_k(fused_scores, relevance_labels, k=5)

            alpha_values.append(alpha)
            metric_values.append(score)

        best_idx = np.argmax(metric_values)
        alpha_empirical = alpha_values[best_idx]
        best_metric = metric_values[best_idx]

        return {
            'alpha_empirical': float(alpha_empirical),
            'best_metric_value': float(best_metric),
            'metric_name': metric,
            'alpha_curve': [float(a) for a in alpha_values],
            'metric_curve': [float(m) for m in metric_values]
        }

    def validate_theory_vs_empirical(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        relevance_labels: np.ndarray,
        metric: str = 'ndcg'
    ) -> Dict[str, Any]:
        """验证理论预测与经验最优的一致性"""
        info_result = self.compute_optimal_alpha_information_theory(
            bm25_scores, dense_scores, relevance_labels
        )
        bayes_result = self.compute_optimal_alpha_bayesian(
            bm25_scores, dense_scores, relevance_labels
        )
        empirical_result = self.compute_optimal_alpha_empirical(
            bm25_scores, dense_scores, relevance_labels, metric=metric
        )

        alpha_theory = info_result['alpha_theory']
        alpha_bayesian = bayes_result['alpha_bayesian']
        alpha_empirical = empirical_result['alpha_empirical']

        absolute_error_info = abs(alpha_theory - alpha_empirical)
        absolute_error_bayes = abs(alpha_bayesian - alpha_empirical)

        bm25_normed = self._normalize_scores(bm25_scores)
        dense_normed = self._normalize_scores(dense_scores)
        fused_theory = (1 - alpha_theory) * dense_normed + alpha_theory * bm25_normed

        if metric == 'ndcg':
            metric_at_theory = self._compute_ndcg(fused_theory, relevance_labels, k=5)
        else:
            metric_at_theory = self._compute_recall_at_k(fused_theory, relevance_labels, k=5)

        metric_at_empirical = empirical_result['best_metric_value']

        if metric_at_empirical > 1e-6:
            performance_gap = (metric_at_empirical - metric_at_theory) / metric_at_empirical * 100
        else:
            performance_gap = 0.0

        return {
            'alpha_theory': float(alpha_theory),
            'alpha_bayesian': float(alpha_bayesian),
            'alpha_empirical': float(alpha_empirical),
            'absolute_error_info': float(absolute_error_info),
            'absolute_error_bayes': float(absolute_error_bayes),
            'metric_at_theory': float(metric_at_theory),
            'metric_at_empirical': float(metric_at_empirical),
            'performance_gap_percent': float(performance_gap),
            'info_theory_params': info_result,
            'bayesian_params': bayes_result,
            'empirical_curve': empirical_result
        }

    def cross_domain_analysis(
        self,
        domain_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        metric: str = 'ndcg'
    ) -> Dict[str, Any]:
        """跨域分析：验证理论公式在不同域的稳定性"""
        results = {}
        theory_alphas = []
        empirical_alphas = []

        for domain_name, (bm25_scores, dense_scores, rel_labels) in domain_data.items():
            validation = self.validate_theory_vs_empirical(
                bm25_scores, dense_scores, rel_labels, metric=metric
            )
            results[domain_name] = validation
            theory_alphas.append(validation['alpha_theory'])
            empirical_alphas.append(validation['alpha_empirical'])

        theory_alphas = np.array(theory_alphas)
        empirical_alphas = np.array(empirical_alphas)

        if len(theory_alphas) > 1:
            corr, p_value = stats.pearsonr(theory_alphas, empirical_alphas)
        else:
            corr, p_value = 0.0, 1.0

        mse = np.mean((theory_alphas - empirical_alphas) ** 2)
        mae = np.mean(np.abs(theory_alphas - empirical_alphas))

        return {
            'domain_results': results,
            'cross_domain_stats': {
                'correlation': float(corr),
                'p_value': float(p_value),
                'mse': float(mse),
                'mae': float(mae),
                'mean_theory_alpha': float(np.mean(theory_alphas)),
                'std_theory_alpha': float(np.std(theory_alphas)),
                'mean_empirical_alpha': float(np.mean(empirical_alphas)),
                'std_empirical_alpha': float(np.std(empirical_alphas)),
                'theory_alphas': [float(a) for a in theory_alphas],
                'empirical_alphas': [float(a) for a in empirical_alphas]
            }
        }

    def export_latex_formula(self) -> str:
        """导出 LaTeX 格式的理论公式"""
        return r"""
% Information-Theoretic Optimal Weight Formula
\begin{equation}
\alpha^* = \frac{\sigma^2_{\text{Dense}} \cdot \rho_{\text{BM25}}}{\sigma^2_{\text{BM25}} \cdot \rho_{\text{Dense}} + \sigma^2_{\text{Dense}} \cdot \rho_{\text{BM25}}}
\end{equation}

% Bayesian Framework
\begin{equation}
P(R=1 \mid S_{\text{BM25}}, S_{\text{Dense}}) \propto P(S_{\text{BM25}} \mid R=1)^\alpha \times P(S_{\text{Dense}} \mid R=1)^{1-\alpha}
\end{equation}

% Fusion Score
\begin{equation}
S_{\text{fusion}} = (1 - \alpha) \cdot S_{\text{Dense}} + \alpha \cdot S_{\text{BM25}}
\end{equation}
"""

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.array(scores, dtype=float)
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val > 1e-10:
            return (scores - min_val) / (max_val - min_val)
        return np.zeros_like(scores)

    def _compute_ndcg(self, scores: np.ndarray, relevance: np.ndarray, k: int = 5) -> float:
        sorted_indices = np.argsort(scores)[::-1][:k]
        sorted_rel = relevance[sorted_indices]
        dcg = np.sum(sorted_rel / np.log2(np.arange(2, len(sorted_rel) + 2)))
        ideal_rel = np.sort(relevance)[::-1][:k]
        idcg = np.sum(ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2)))
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(self, scores: np.ndarray, relevance: np.ndarray) -> float:
        sorted_indices = np.argsort(scores)[::-1]
        sorted_rel = relevance[sorted_indices]
        n_relevant = np.sum(relevance)
        if n_relevant == 0:
            return 0.0
        precisions = []
        n_hits = 0
        for i, rel in enumerate(sorted_rel):
            if rel == 1:
                n_hits += 1
                precisions.append(n_hits / (i + 1))
        return np.mean(precisions) if precisions else 0.0

    def _compute_recall_at_k(self, scores: np.ndarray, relevance: np.ndarray, k: int = 5) -> float:
        sorted_indices = np.argsort(scores)[::-1][:k]
        retrieved_rel = relevance[sorted_indices]
        n_relevant_total = np.sum(relevance)
        if n_relevant_total == 0:
            return 0.0
        return np.sum(retrieved_rel) / n_relevant_total


def run_alpha_theory_experiment(
    data_path: str,
    output_dir: str,
    domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """运行完整的 α 理论验证实验"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if domains is not None:
        data = {k: v for k, v in data.items() if k in domains}

    domain_data = {}
    for domain, values in data.items():
        domain_data[domain] = (
            np.array(values['bm25_scores']),
            np.array(values['dense_scores']),
            np.array(values['relevance_labels'])
        )

    analyzer = AlphaTheoryAnalyzer()

    print("Running cross-domain analysis...")
    cross_domain_results = analyzer.cross_domain_analysis(domain_data, metric='ndcg')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    formula_path = output_path / 'theory_formula.tex'
    with open(formula_path, 'w', encoding='utf-8') as f:
        f.write(analyzer.export_latex_formula())

    result_path = output_path / 'alpha_theory_results.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(cross_domain_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir}")

    stats_data = cross_domain_results['cross_domain_stats']
    print(f"\n=== Cross-Domain Statistics ===")
    print(f"Correlation (Theory vs Empirical): {stats_data['correlation']:.4f}")
    print(f"Mean Absolute Error: {stats_data['mae']:.4f}")
    print(f"Mean Theory α: {stats_data['mean_theory_alpha']:.4f} ± {stats_data['std_theory_alpha']:.4f}")
    print(f"Mean Empirical α: {stats_data['mean_empirical_alpha']:.4f} ± {stats_data['std_empirical_alpha']:.4f}")

    return {
        'cross_domain_results': cross_domain_results,
        'result_path': str(result_path),
        'formula_path': str(formula_path)
    }
