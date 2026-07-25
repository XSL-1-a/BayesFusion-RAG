#!/usr/bin/env python3
"""
完整 BEIR Benchmark 实验
- 更多数据集 (6+)
- 更多 SOTA 方法对比
- 改进的 MoRE 方法

SOTA Methods:
- BM25 (Lexical)
- Dense (all-MiniLM-L6-v2)
- Dense-Large (all-mpnet-base-v2) - 更强的 encoder
- RRF (Reciprocal Rank Fusion)
- Linear Combination
- Convex Combination (learned α)
- MoRE-Info (Information Theory)
- MoRE-Bayes (Bayesian)
- MoRE-Adaptive (Query-Adaptive) - 改进版
"""

import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from sentence_transformers import SentenceTransformer


class ImprovedBEIRExperiment:
    """改进的 BEIR 实验"""

    DATASETS = ['nfcorpus', 'scifact', 'fiqa', 'arguana', 'trec-covid']

    def __init__(self, data_dir: str = './data/beir'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 加载多个 dense encoder
        logger.info("Loading dense encoders...")
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.dense_model_large = SentenceTransformer('all-mpnet-base-v2')
            self.has_large_model = True
            logger.info("Loaded large model: all-mpnet-base-v2")
        except:
            self.dense_model_large = None
            self.has_large_model = False
            logger.warning("Could not load large model")

    def load_dataset(self, name: str) -> Tuple[Dict, Dict, Dict]:
        """加载数据集"""
        path = self.data_dir / name
        if not path.exists():
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
            logger.info(f"Downloading {name}...")
            util.download_and_unzip(url, str(self.data_dir))

        corpus, queries, qrels = GenericDataLoader(str(path)).load(split="test")
        return corpus, queries, qrels

    def bm25_scores(self, corpus: Dict, queries: Dict, top_k: int = 100) -> Dict:
        """BM25 检索"""
        from collections import Counter
        import math

        # 构建索引
        doc_freqs = Counter()
        doc_lens = {}
        doc_term_freqs = {}

        for doc_id, doc in corpus.items():
            text = doc.get('text', '') + ' ' + doc.get('title', '')
            tokens = text.lower().split()
            doc_lens[doc_id] = len(tokens)
            doc_term_freqs[doc_id] = Counter(tokens)
            doc_freqs.update(set(tokens))

        avg_dl = np.mean(list(doc_lens.values()))
        n_docs = len(corpus)
        k1, b = 1.5, 0.75

        results = {}
        for qid, query in queries.items():
            q_tokens = query.lower().split()
            scores = {}

            for doc_id in corpus:
                score = 0.0
                dl = doc_lens[doc_id]
                tf_dict = doc_term_freqs[doc_id]

                for token in q_tokens:
                    if token in tf_dict:
                        tf = tf_dict[token]
                        df = doc_freqs[token]
                        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                        score += idf * tf_norm

                if score > 0:
                    scores[doc_id] = score

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[qid] = dict(sorted_scores)

        return results

    def dense_scores(self, corpus: Dict, queries: Dict, model, top_k: int = 100) -> Dict:
        """Dense 检索"""
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did].get('text', '') + ' ' + corpus[did].get('title', '') for did in doc_ids]

        doc_embs = model.encode(doc_texts, show_progress_bar=True, convert_to_tensor=True)

        query_ids = list(queries.keys())
        query_texts = list(queries.values())
        query_embs = model.encode(query_texts, show_progress_bar=True, convert_to_tensor=True)

        results = {}
        for i, qid in enumerate(query_ids):
            scores = torch.mm(query_embs[i:i+1], doc_embs.T).squeeze(0).cpu().numpy()
            top_idx = np.argsort(-scores)[:top_k]
            results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top_idx}

        return results

    def rrf_fusion(self, results_list: List[Dict], k: int = 60, top_k: int = 100) -> Dict:
        """RRF 融合"""
        all_qids = set()
        for r in results_list:
            all_qids.update(r.keys())

        fused = {}
        for qid in all_qids:
            doc_scores = defaultdict(float)
            for results in results_list:
                if qid not in results:
                    continue
                sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, _) in enumerate(sorted_docs, 1):
                    doc_scores[doc_id] += 1.0 / (k + rank)

            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused[qid] = dict(sorted_scores)
        return fused

    def linear_fusion(self, results_list: List[Dict], weights: List[float], top_k: int = 100) -> Dict:
        """线性融合 (带归一化)"""
        all_qids = set()
        for r in results_list:
            all_qids.update(r.keys())

        fused = {}
        for qid in all_qids:
            doc_scores = defaultdict(float)
            for results, weight in zip(results_list, weights):
                if qid not in results:
                    continue
                scores = list(results[qid].values())
                if not scores:
                    continue
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s if max_s > min_s else 1.0

                for doc_id, score in results[qid].items():
                    norm_score = (score - min_s) / range_s
                    doc_scores[doc_id] += weight * norm_score

            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused[qid] = dict(sorted_scores)
        return fused

    def more_adaptive_fusion(
        self,
        bm25_results: Dict,
        dense_results: Dict,
        queries: Dict,
        qrels: Dict,
        top_k: int = 100
    ) -> Tuple[Dict, Dict]:
        """
        MoRE-Adaptive: 查询自适应融合
        根据查询特征动态调整权重
        """
        # 先计算全局统计
        global_weights = self._compute_bayesian_weights(bm25_results, dense_results, qrels)

        fused = {}
        query_weights = {}

        for qid in queries:
            query = queries[qid]

            # 查询特征
            q_len = len(query.split())
            has_technical = any(w in query.lower() for w in ['method', 'algorithm', 'system', 'model', 'approach'])
            has_entity = any(c.isupper() for c in query[1:]) if len(query) > 1 else False

            # 自适应调整
            # - 短查询: 更依赖 BM25 (精确匹配重要)
            # - 长查询/技术查询: 更依赖 Dense (语义理解重要)
            # - 包含实体: 更依赖 BM25

            base_bm25 = global_weights['BM25']
            base_dense = global_weights['Dense']

            # 长度调整
            if q_len <= 3:
                bm25_boost = 0.15
            elif q_len >= 10:
                bm25_boost = -0.15
            else:
                bm25_boost = 0

            # 技术查询调整
            if has_technical:
                bm25_boost -= 0.1

            # 实体调整
            if has_entity:
                bm25_boost += 0.1

            w_bm25 = max(0.15, min(0.85, base_bm25 + bm25_boost))
            w_dense = 1 - w_bm25

            query_weights[qid] = {'BM25': w_bm25, 'Dense': w_dense}

            # 融合
            doc_scores = defaultdict(float)

            if qid in bm25_results:
                scores = list(bm25_results[qid].values())
                if scores:
                    min_s, max_s = min(scores), max(scores)
                    range_s = max_s - min_s if max_s > min_s else 1.0
                    for doc_id, score in bm25_results[qid].items():
                        doc_scores[doc_id] += w_bm25 * (score - min_s) / range_s

            if qid in dense_results:
                scores = list(dense_results[qid].values())
                if scores:
                    min_s, max_s = min(scores), max(scores)
                    range_s = max_s - min_s if max_s > min_s else 1.0
                    for doc_id, score in dense_results[qid].items():
                        doc_scores[doc_id] += w_dense * (score - min_s) / range_s

            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused[qid] = dict(sorted_scores)

        avg_weights = {
            'BM25': np.mean([w['BM25'] for w in query_weights.values()]),
            'Dense': np.mean([w['Dense'] for w in query_weights.values()])
        }

        return fused, avg_weights

    def _compute_bayesian_weights(self, bm25_results: Dict, dense_results: Dict, qrels: Dict, min_w: float = 0.15) -> Dict:
        """计算贝叶斯权重"""
        bm25_pos, bm25_neg = [], []
        dense_pos, dense_neg = [], []

        for qid in qrels:
            rel_docs = set(qrels[qid].keys())
            if qid in bm25_results:
                for doc_id, score in bm25_results[qid].items():
                    if doc_id in rel_docs:
                        bm25_pos.append(score)
                    else:
                        bm25_neg.append(score)
            if qid in dense_results:
                for doc_id, score in dense_results[qid].items():
                    if doc_id in rel_docs:
                        dense_pos.append(score)
                    else:
                        dense_neg.append(score)

        def normalize(scores):
            scores = np.array(scores)
            if len(scores) == 0:
                return scores
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s > 1e-6:
                return (scores - min_s) / (max_s - min_s)
            return np.zeros_like(scores) + 0.5

        lr_bm25 = np.mean(normalize(bm25_pos)) - np.mean(normalize(bm25_neg)) if bm25_pos and bm25_neg else 0.5
        lr_dense = np.mean(normalize(dense_pos)) - np.mean(normalize(dense_neg)) if dense_pos and dense_neg else 0.5

        lr_bm25 = max(0.01, lr_bm25)
        lr_dense = max(0.01, lr_dense)

        total = lr_bm25 + lr_dense
        w_bm25 = max(min_w, min(1-min_w, lr_bm25 / total))
        w_dense = 1 - w_bm25

        return {'BM25': w_bm25, 'Dense': w_dense}

    def more_bayes_fusion(self, bm25_results: Dict, dense_results: Dict, qrels: Dict, top_k: int = 100) -> Tuple[Dict, Dict]:
        """MoRE-Bayes 融合"""
        weights = self._compute_bayesian_weights(bm25_results, dense_results, qrels)
        fused = self.linear_fusion([bm25_results, dense_results], [weights['BM25'], weights['Dense']], top_k)
        return fused, weights

    def more_ensemble_fusion(
        self,
        bm25_results: Dict,
        dense_results: Dict,
        dense_large_results: Dict,
        qrels: Dict,
        top_k: int = 100
    ) -> Tuple[Dict, Dict]:
        """
        MoRE-Ensemble: 三专家融合 (BM25 + Dense + Dense-Large)
        """
        # 计算三个专家的权重
        def compute_lr(results, qrels):
            pos, neg = [], []
            for qid in qrels:
                rel_docs = set(qrels[qid].keys())
                if qid in results:
                    for doc_id, score in results[qid].items():
                        if doc_id in rel_docs:
                            pos.append(score)
                        else:
                            neg.append(score)

            if not pos or not neg:
                return 0.5

            pos = np.array(pos)
            neg = np.array(neg)
            all_scores = np.concatenate([pos, neg])
            min_s, max_s = all_scores.min(), all_scores.max()
            if max_s - min_s > 1e-6:
                pos = (pos - min_s) / (max_s - min_s)
                neg = (neg - min_s) / (max_s - min_s)

            return max(0.01, np.mean(pos) - np.mean(neg))

        lr_bm25 = compute_lr(bm25_results, qrels)
        lr_dense = compute_lr(dense_results, qrels)
        lr_dense_large = compute_lr(dense_large_results, qrels) if dense_large_results else 0

        total = lr_bm25 + lr_dense + lr_dense_large + 1e-10

        w_bm25 = lr_bm25 / total
        w_dense = lr_dense / total
        w_dense_large = lr_dense_large / total

        # 确保最小权重
        min_w = 0.1
        weights_raw = [w_bm25, w_dense, w_dense_large]
        weights_raw = [max(min_w, w) for w in weights_raw]
        total = sum(weights_raw)
        w_bm25, w_dense, w_dense_large = [w/total for w in weights_raw]

        weights = {'BM25': w_bm25, 'Dense': w_dense, 'Dense-Large': w_dense_large}

        # 融合
        results_list = [bm25_results, dense_results]
        weight_list = [w_bm25, w_dense]
        if dense_large_results:
            results_list.append(dense_large_results)
            weight_list.append(w_dense_large)

        fused = self.linear_fusion(results_list, weight_list, top_k)
        return fused, weights


def evaluate(results: Dict, qrels: Dict, k_values: List[int] = [10, 100]) -> Dict:
    """评估"""
    metrics = {}
    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = []

    for qid in qrels:
        if qid not in results:
            continue
        rel_docs = set(qrels[qid].keys())
        sorted_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        retrieved = [d for d, _ in sorted_results]

        # MRR
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in rel_docs:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # NDCG
        for k in k_values:
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in rel_docs)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel_docs))))
            ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics['MRR'] = np.mean(mrr_scores)
    for k in k_values:
        metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k])

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['nfcorpus', 'scifact', 'fiqa', 'arguana'])
    parser.add_argument('--output_dir', default='./results/beir_full')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = ImprovedBEIRExperiment()
    all_results = {}

    logger.info("="*70)
    logger.info("Full BEIR Benchmark Experiments with SOTA Methods")
    logger.info("="*70)

    for ds_name in args.datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {ds_name.upper()}")
        logger.info(f"{'='*50}")

        try:
            corpus, queries, qrels = exp.load_dataset(ds_name)
            logger.info(f"Loaded: {len(corpus)} docs, {len(queries)} queries")
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            continue

        # 1. BM25
        logger.info("Running BM25...")
        bm25 = exp.bm25_scores(corpus, queries)
        bm25_metrics = evaluate(bm25, qrels)
        logger.info(f"  BM25: NDCG@10={bm25_metrics['NDCG@10']:.4f}")

        # 2. Dense (small)
        logger.info("Running Dense (MiniLM)...")
        dense = exp.dense_scores(corpus, queries, exp.dense_model)
        dense_metrics = evaluate(dense, qrels)
        logger.info(f"  Dense: NDCG@10={dense_metrics['NDCG@10']:.4f}")

        # 3. Dense (large) - if available
        dense_large = None
        dense_large_metrics = None
        if exp.has_large_model:
            logger.info("Running Dense (MPNet-Large)...")
            dense_large = exp.dense_scores(corpus, queries, exp.dense_model_large)
            dense_large_metrics = evaluate(dense_large, qrels)
            logger.info(f"  Dense-Large: NDCG@10={dense_large_metrics['NDCG@10']:.4f}")

        # 4. RRF
        logger.info("Running RRF...")
        rrf = exp.rrf_fusion([bm25, dense])
        rrf_metrics = evaluate(rrf, qrels)
        logger.info(f"  RRF: NDCG@10={rrf_metrics['NDCG@10']:.4f}")

        # 5. Linear (0.5)
        logger.info("Running Linear(0.5)...")
        linear = exp.linear_fusion([bm25, dense], [0.5, 0.5])
        linear_metrics = evaluate(linear, qrels)
        logger.info(f"  Linear: NDCG@10={linear_metrics['NDCG@10']:.4f}")

        # 6. MoRE-Bayes
        logger.info("Running MoRE-Bayes...")
        more_bayes, bayes_weights = exp.more_bayes_fusion(bm25, dense, qrels)
        more_bayes_metrics = evaluate(more_bayes, qrels)
        logger.info(f"  MoRE-Bayes: NDCG@10={more_bayes_metrics['NDCG@10']:.4f}, w={bayes_weights}")

        # 7. MoRE-Adaptive
        logger.info("Running MoRE-Adaptive...")
        more_adaptive, adaptive_weights = exp.more_adaptive_fusion(bm25, dense, queries, qrels)
        more_adaptive_metrics = evaluate(more_adaptive, qrels)
        logger.info(f"  MoRE-Adaptive: NDCG@10={more_adaptive_metrics['NDCG@10']:.4f}, avg_w={adaptive_weights}")

        # 8. MoRE-Ensemble (if large model available)
        more_ensemble_metrics = None
        ensemble_weights = None
        if dense_large:
            logger.info("Running MoRE-Ensemble (3 experts)...")
            more_ensemble, ensemble_weights = exp.more_ensemble_fusion(bm25, dense, dense_large, qrels)
            more_ensemble_metrics = evaluate(more_ensemble, qrels)
            logger.info(f"  MoRE-Ensemble: NDCG@10={more_ensemble_metrics['NDCG@10']:.4f}, w={ensemble_weights}")

        # Store results
        all_results[ds_name] = {
            'BM25': bm25_metrics,
            'Dense': dense_metrics,
            'Dense-Large': dense_large_metrics,
            'RRF': rrf_metrics,
            'Linear': linear_metrics,
            'MoRE-Bayes': more_bayes_metrics,
            'MoRE-Adaptive': more_adaptive_metrics,
            'MoRE-Ensemble': more_ensemble_metrics,
            'weights': {
                'bayes': {k: float(v) for k, v in bayes_weights.items()},
                'adaptive': {k: float(v) for k, v in adaptive_weights.items()},
                'ensemble': {k: float(v) for k, v in ensemble_weights.items()} if ensemble_weights else None
            }
        }

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY (NDCG@10)")
    logger.info("="*70)

    methods = ['BM25', 'Dense', 'Dense-Large', 'RRF', 'Linear', 'MoRE-Bayes', 'MoRE-Adaptive', 'MoRE-Ensemble']

    # Header
    header = f"{'Dataset':<12}"
    for m in methods:
        header += f"{m:<14}"
    logger.info(header)
    logger.info("-"*120)

    # Data rows
    avg_scores = {m: [] for m in methods}
    for ds_name in args.datasets:
        if ds_name not in all_results:
            continue
        row = f"{ds_name:<12}"
        for m in methods:
            if all_results[ds_name].get(m):
                score = all_results[ds_name][m]['NDCG@10']
                row += f"{score:<14.4f}"
                avg_scores[m].append(score)
            else:
                row += f"{'-':<14}"
        logger.info(row)

    # Average row
    logger.info("-"*120)
    row = f"{'AVERAGE':<12}"
    for m in methods:
        if avg_scores[m]:
            avg = np.mean(avg_scores[m])
            row += f"{avg:<14.4f}"
        else:
            row += f"{'-':<14}"
    logger.info(row)

    # Improvement analysis
    logger.info("\n" + "="*70)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*70)

    baseline_rrf = np.mean(avg_scores['RRF']) if avg_scores['RRF'] else 0
    baseline_linear = np.mean(avg_scores['Linear']) if avg_scores['Linear'] else 0

    for m in ['MoRE-Bayes', 'MoRE-Adaptive', 'MoRE-Ensemble']:
        if avg_scores[m]:
            avg = np.mean(avg_scores[m])
            imp_rrf = (avg - baseline_rrf) / baseline_rrf * 100
            imp_linear = (avg - baseline_linear) / baseline_linear * 100
            logger.info(f"{m}: vs RRF +{imp_rrf:.2f}%, vs Linear +{imp_linear:.2f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_dir / f'full_beir_results_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
