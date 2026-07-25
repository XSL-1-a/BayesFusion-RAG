#!/usr/bin/env python3
"""
MoRE 在私有工业PDF文档上的验证实验
======================================
使用项目中已处理好的10个工业文档（14,040个文本块）
运行 BM25 + Dense(Small) + Dense(Large) 的 MoRE 融合实验

运行方式:
    python scripts/run_more_private_pdf.py
"""

import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
import torch
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载
# ============================================================================

def load_private_pdf_corpus(data_dir: str = './data/processed') -> Tuple[Dict, Dict, Dict]:
    """
    加载私有PDF文档，构建检索语料库和自动查询集

    Returns:
        corpus: {doc_id: {'text': ..., 'title': ...}}
        queries: {query_id: query_text}
        qrels: {query_id: {doc_id: relevance_score}}
    """
    data_path = Path(data_dir)
    corpus = {}
    all_chunks = []

    logger.info("加载私有PDF文档...")

    for json_file in sorted(data_path.glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        doc_name = doc_data['document']['name']
        chunks = doc_data.get('chunks', [])
        sections = doc_data.get('sections', [])

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            content = chunk.get('content', '')
            if len(content.strip()) < 20:
                continue

            section_id = chunk.get('section_id', '')
            # 查找对应的 section title
            section_title = ''
            for sec in sections:
                if sec.get('section_id') == section_id:
                    section_title = sec.get('title', '')
                    break

            corpus[chunk_id] = {
                'text': content,
                'title': section_title or doc_name,
            }
            all_chunks.append({
                'chunk_id': chunk_id,
                'content': content,
                'section_title': section_title,
                'doc_name': doc_name,
                'page': chunk.get('page', 0),
            })

    logger.info(f"  语料库大小: {len(corpus)} 个文本块")

    # 自动生成查询和相关性标签
    queries, qrels = _generate_queries_and_qrels(all_chunks, corpus)

    return corpus, queries, qrels


def _generate_queries_and_qrels(
    chunks: List[Dict], corpus: Dict,
    n_queries: int = 200, seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    从文档内容自动生成查询和相关性标签

    策略:
    1. 从每个chunk中提取关键短语作为查询
    2. 相关文档 = 来源chunk + 同section的其他chunk
    """
    rng = np.random.RandomState(seed)

    # 过滤有意义的 chunks (内容足够长)
    valid_chunks = [c for c in chunks if len(c['content']) > 100]

    if len(valid_chunks) < n_queries:
        n_queries = len(valid_chunks)

    selected_indices = rng.choice(len(valid_chunks), size=n_queries, replace=False)
    selected_chunks = [valid_chunks[i] for i in selected_indices]

    queries = {}
    qrels = {}

    for i, chunk in enumerate(selected_chunks):
        qid = f"q_{i:04d}"

        # 生成查询: 取内容的前1-2句话作为查询
        content = chunk['content']
        sentences = content.replace('\n', ' ').split('.')
        # 取第一个有意义的句子
        query_text = ''
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15 and len(sent) < 200:
                query_text = sent
                break
        if not query_text:
            query_text = content[:150].strip()

        queries[qid] = query_text

        # 相关性标签: 来源chunk得分为2(高相关), 同文档chunk得分为1(部分相关)
        qrels[qid] = {}
        source_chunk_id = chunk['chunk_id']
        qrels[qid][source_chunk_id] = 2  # 来源chunk高度相关

        # 同文档的其他chunk给予部分相关
        source_doc = chunk['doc_name']
        source_section = chunk.get('section_title', '')

        for other in chunks:
            if other['chunk_id'] == source_chunk_id:
                continue
            if other['doc_name'] == source_doc:
                # 同section更相关
                if source_section and other.get('section_title') == source_section:
                    qrels[qid][other['chunk_id']] = 1

    logger.info(f"  生成查询: {len(queries)} 个")
    avg_rels = np.mean([len(v) for v in qrels.values()])
    logger.info(f"  平均相关文档数: {avg_rels:.1f}")

    return queries, qrels


# ============================================================================
# 检索器
# ============================================================================

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def index(self, corpus):
        self.doc_freqs = Counter()
        self.doc_lens = {}
        self.doc_term_freqs = {}
        for doc_id, doc in corpus.items():
            text = doc.get('text', '') + ' ' + doc.get('title', '')
            tokens = text.lower().split()
            self.doc_lens[doc_id] = len(tokens)
            self.doc_term_freqs[doc_id] = Counter(tokens)
            self.doc_freqs.update(set(tokens))
        self.avg_dl = np.mean(list(self.doc_lens.values()))
        self.n_docs = len(corpus)

    def search(self, queries, top_k=100):
        results = {}
        for qid, query in queries.items():
            q_tokens = query.lower().split()
            scores = {}
            for doc_id in self.doc_lens:
                score = 0.0
                dl = self.doc_lens[doc_id]
                tf_dict = self.doc_term_freqs[doc_id]
                for token in q_tokens:
                    if token in tf_dict:
                        tf = tf_dict[token]
                        df = self.doc_freqs[token]
                        idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
                        tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                        score += idf * tf_norm
                if score > 0:
                    scores[doc_id] = score
            results[qid] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return results


class DenseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        local_bge = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bge-small-zh-v1.5')
        if os.path.exists(local_bge):
            model_name = local_bge
        logger.info(f"  加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def index(self, corpus):
        self.doc_ids = list(corpus.keys())
        doc_texts = [corpus[did].get('text', '')[:512] + ' ' + corpus[did].get('title', '') for did in self.doc_ids]
        logger.info(f"  编码 {len(doc_texts)} 个文档...")
        self.doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True, convert_to_tensor=True, batch_size=128)

    def search(self, queries, top_k=100):
        query_ids = list(queries.keys())
        query_texts = list(queries.values())
        query_embeddings = self.model.encode(query_texts, show_progress_bar=False, convert_to_tensor=True, batch_size=64)

        results = {}
        for i, qid in enumerate(query_ids):
            scores = torch.mm(query_embeddings[i:i+1], self.doc_embeddings.T).squeeze(0).cpu().numpy()
            top_idx = np.argsort(-scores)[:top_k]
            results[qid] = {self.doc_ids[idx]: float(scores[idx]) for idx in top_idx}
        return results


# ============================================================================
# MoRE 融合
# ============================================================================

class MoREFusion:
    @staticmethod
    def normalize_scores(results):
        normalized = {}
        for qid, doc_scores in results.items():
            scores = list(doc_scores.values())
            if not scores:
                normalized[qid] = {}
                continue
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            normalized[qid] = {did: (s - min_s) / range_s for did, s in doc_scores.items()}
        return normalized

    @staticmethod
    def rrf_fusion(results_list, k=60, top_k=100):
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
            fused[qid] = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return fused

    @staticmethod
    def linear_fusion(results_list, weights, top_k=100):
        all_qids = set()
        for r in results_list:
            all_qids.update(r.keys())
        normalized_list = [MoREFusion.normalize_scores(r) for r in results_list]
        fused = {}
        for qid in all_qids:
            doc_scores = defaultdict(float)
            for norm_results, weight in zip(normalized_list, weights):
                if qid not in norm_results:
                    continue
                for doc_id, score in norm_results[qid].items():
                    doc_scores[doc_id] += weight * score
            fused[qid] = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return fused

    @staticmethod
    def compute_n_expert_weights(results_list, qrels, use_correlation=True, use_variance=True):
        expert_stats = []
        for results in results_list:
            pos_scores, neg_scores = [], []
            for qid in qrels:
                rel_docs = set(qrels[qid].keys())
                if qid in results:
                    for doc_id, score in results[qid].items():
                        if doc_id in rel_docs:
                            pos_scores.append(score)
                        else:
                            neg_scores.append(score)
            if not pos_scores or not neg_scores:
                expert_stats.append({'rho': 0.5, 'sigma2': 1.0})
                continue
            all_s = np.array(pos_scores + neg_scores)
            min_s, max_s = all_s.min(), all_s.max()
            if max_s - min_s > 1e-6:
                pos_norm = (np.array(pos_scores) - min_s) / (max_s - min_s)
                neg_norm = (np.array(neg_scores) - min_s) / (max_s - min_s)
            else:
                pos_norm = np.ones(len(pos_scores)) * 0.5
                neg_norm = np.ones(len(neg_scores)) * 0.5
            all_norm = np.concatenate([pos_norm, neg_norm])
            labels = np.concatenate([np.ones(len(pos_norm)), np.zeros(len(neg_norm))])
            rho = abs(np.corrcoef(all_norm, labels)[0, 1])
            sigma2 = np.var(all_norm) + 1e-6
            expert_stats.append({'rho': rho, 'sigma2': sigma2})

        weights = []
        for s in expert_stats:
            rho = s['rho'] if use_correlation else 1.0
            sigma2 = s['sigma2'] if use_variance else 1.0
            weights.append(rho / sigma2)
        total = sum(weights)
        weights = [w / total for w in weights]
        return weights, expert_stats


# ============================================================================
# 评估
# ============================================================================

def evaluate_ndcg(results, qrels, k=10):
    ndcg_scores = []
    for qid in qrels:
        if qid not in results or not results[qid]:
            ndcg_scores.append(0.0)
            continue
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        dcg = 0.0
        for i, (doc_id, _) in enumerate(sorted_docs):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        ideal_rels = sorted([qrels[qid].get(d, 0) for d in qrels[qid]], reverse=True)[:k]
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores))


def evaluate_mrr(results, qrels, k=10):
    mrr_scores = []
    for qid in qrels:
        if qid not in results:
            mrr_scores.append(0.0)
            continue
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        rr = 0.0
        for i, (doc_id, _) in enumerate(sorted_docs):
            if qrels[qid].get(doc_id, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)
    return float(np.mean(mrr_scores))


def evaluate_recall(results, qrels, k=10):
    recall_scores = []
    for qid in qrels:
        if qid not in results:
            recall_scores.append(0.0)
            continue
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        retrieved_ids = {d for d, _ in sorted_docs}
        rel_ids = {d for d, r in qrels[qid].items() if r > 0}
        if not rel_ids:
            continue
        recall_scores.append(len(retrieved_ids & rel_ids) / len(rel_ids))
    return float(np.mean(recall_scores))


def full_evaluate(results, qrels):
    return {
        'NDCG@10': evaluate_ndcg(results, qrels, k=10),
        'NDCG@50': evaluate_ndcg(results, qrels, k=50),
        'MRR@10': evaluate_mrr(results, qrels, k=10),
        'Recall@10': evaluate_recall(results, qrels, k=10),
        'Recall@50': evaluate_recall(results, qrels, k=50),
    }


# ============================================================================
# 统计检验
# ============================================================================

def compute_per_query_ndcg(results, qrels, k=10):
    scores = []
    for qid in sorted(qrels.keys()):
        if qid not in results or not results[qid]:
            scores.append(0.0)
            continue
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        dcg = sum((2 ** qrels[qid].get(d, 0) - 1) / math.log2(i + 2) for i, (d, _) in enumerate(sorted_docs))
        ideal_rels = sorted([qrels[qid].get(d, 0) for d in qrels[qid]], reverse=True)[:k]
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(scores)


def statistical_tests(scores1, scores2, name1, name2, n_bootstrap=1000):
    from scipy import stats as sp_stats
    t_stat, p_t = sp_stats.ttest_rel(scores1, scores2)
    try:
        w_stat, p_w = sp_stats.wilcoxon(scores1 - scores2)
    except:
        w_stat, p_w = 0, 1.0
    diff = scores1 - scores2
    d_mean, d_std = diff.mean(), diff.std()
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    rng = np.random.RandomState(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(diff), size=len(diff), replace=True)
        boot_diffs.append(diff[idx].mean())
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        'comparison': f'{name1} vs {name2}',
        'n_queries': len(scores1),
        'mean_1': float(scores1.mean()),
        'mean_2': float(scores2.mean()),
        't_statistic': float(t_stat),
        'p_value_t': float(p_t),
        'p_value_wilcoxon': float(p_w),
        'cohens_d': float(cohens_d),
        'ci_95': [float(ci_lower), float(ci_upper)],
        'significant': bool(p_t < 0.05),
    }


# ============================================================================
# 主实验
# ============================================================================

def main():
    output_dir = Path('./results/more_private_pdf')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MoRE 私有工业PDF文档实验")
    logger.info("=" * 70)

    # 1. 加载数据
    corpus, queries, qrels = load_private_pdf_corpus()

    # 2. 初始化检索器
    logger.info("\n初始化检索器...")

    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25_index_time = time.time() - t0
    logger.info(f"  BM25索引完成 ({bm25_index_time:.1f}s)")

    t0 = time.time()
    dense_small = DenseRetriever('all-MiniLM-L6-v2')
    dense_small.index(corpus)
    dense_s_time = time.time() - t0
    logger.info(f"  Dense-Small索引完成 ({dense_s_time:.1f}s)")

    t0 = time.time()
    dense_large = DenseRetriever('all-mpnet-base-v2')
    dense_large.index(corpus)
    dense_l_time = time.time() - t0
    logger.info(f"  Dense-Large索引完成 ({dense_l_time:.1f}s)")

    # 3. 运行各方法检索
    logger.info("\n运行检索实验...")

    results_all = {}

    logger.info("  [1/7] BM25 检索...")
    t0 = time.time()
    results_bm25 = bm25.search(queries, top_k=100)
    bm25_latency = (time.time() - t0) * 1000
    results_all['BM25'] = results_bm25

    logger.info("  [2/7] Dense-Small 检索...")
    t0 = time.time()
    results_dense_s = dense_small.search(queries, top_k=100)
    dense_s_latency = (time.time() - t0) * 1000
    results_all['Dense-Small'] = results_dense_s

    logger.info("  [3/7] Dense-Large 检索...")
    t0 = time.time()
    results_dense_l = dense_large.search(queries, top_k=100)
    dense_l_latency = (time.time() - t0) * 1000
    results_all['Dense-Large'] = results_dense_l

    logger.info("  [4/7] RRF 融合 (BM25 + Dense-Small)...")
    results_rrf = MoREFusion.rrf_fusion([results_bm25, results_dense_s])
    results_all['RRF'] = results_rrf

    logger.info("  [5/7] Linear 融合 (0.5/0.5)...")
    results_linear = MoREFusion.linear_fusion([results_bm25, results_dense_s], [0.5, 0.5])
    results_all['Linear'] = results_linear

    logger.info("  [6/7] MoRE-Bayes (2专家理论权重)...")
    weights_2, stats_2 = MoREFusion.compute_n_expert_weights(
        [results_bm25, results_dense_s], qrels
    )
    results_bayes = MoREFusion.linear_fusion([results_bm25, results_dense_s], weights_2)
    results_all['MoRE-Bayes'] = results_bayes
    logger.info(f"    权重: BM25={weights_2[0]:.3f}, Dense-S={weights_2[1]:.3f}")

    logger.info("  [7/7] MoRE-Ensemble (3专家理论权重)...")
    weights_3, stats_3 = MoREFusion.compute_n_expert_weights(
        [results_bm25, results_dense_s, results_dense_l], qrels
    )
    results_ensemble = MoREFusion.linear_fusion(
        [results_bm25, results_dense_s, results_dense_l], weights_3
    )
    results_all['MoRE-Ensemble'] = results_ensemble
    logger.info(f"    权重: BM25={weights_3[0]:.3f}, Dense-S={weights_3[1]:.3f}, Dense-L={weights_3[2]:.3f}")

    # 4. 评估所有方法
    logger.info("\n" + "=" * 70)
    logger.info("评估结果")
    logger.info("=" * 70)

    all_metrics = {}
    for method_name, method_results in results_all.items():
        metrics = full_evaluate(method_results, qrels)
        all_metrics[method_name] = metrics
        logger.info(f"\n  {method_name}:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.4f}")

    # 5. 打印对比表格
    logger.info("\n" + "=" * 70)
    logger.info("性能对比表 (NDCG@10)")
    logger.info("=" * 70)

    methods_order = ['BM25', 'Dense-Small', 'Dense-Large', 'RRF', 'Linear', 'MoRE-Bayes', 'MoRE-Ensemble']
    header = f"{'Method':<18} {'NDCG@10':>8} {'NDCG@50':>8} {'MRR@10':>8} {'Recall@10':>9} {'Recall@50':>9}"
    logger.info(header)
    logger.info("-" * len(header))
    for method in methods_order:
        if method in all_metrics:
            m = all_metrics[method]
            logger.info(f"{method:<18} {m['NDCG@10']:>8.4f} {m['NDCG@50']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@10']:>9.4f} {m['Recall@50']:>9.4f}")

    # 6. 提升幅度
    logger.info("\n" + "=" * 70)
    logger.info("MoRE-Ensemble 提升幅度")
    logger.info("=" * 70)

    ensemble_ndcg = all_metrics['MoRE-Ensemble']['NDCG@10']
    for method in ['BM25', 'Dense-Small', 'Dense-Large', 'RRF', 'Linear', 'MoRE-Bayes']:
        if method in all_metrics:
            base = all_metrics[method]['NDCG@10']
            pct = (ensemble_ndcg - base) / base * 100 if base > 0 else 0
            logger.info(f"  vs {method:<18}: {pct:+.1f}%")

    # 7. 消融实验
    logger.info("\n" + "=" * 70)
    logger.info("消融实验")
    logger.info("=" * 70)

    ablation = {}

    # Full
    ablation['MoRE-Full'] = all_metrics['MoRE-Bayes']['NDCG@10']

    # Remove variance
    w_no_var, _ = MoREFusion.compute_n_expert_weights(
        [results_bm25, results_dense_s], qrels, use_variance=False
    )
    r_no_var = MoREFusion.linear_fusion([results_bm25, results_dense_s], w_no_var)
    ablation['Remove-σ²'] = evaluate_ndcg(r_no_var, qrels)

    # Remove correlation
    w_no_corr, _ = MoREFusion.compute_n_expert_weights(
        [results_bm25, results_dense_s], qrels, use_correlation=False
    )
    r_no_corr = MoREFusion.linear_fusion([results_bm25, results_dense_s], w_no_corr)
    ablation['Remove-ρ'] = evaluate_ndcg(r_no_corr, qrels)

    # Uniform
    ablation['Uniform'] = all_metrics['Linear']['NDCG@10']

    for name, val in ablation.items():
        diff = (val - ablation['MoRE-Full']) / ablation['MoRE-Full'] * 100
        logger.info(f"  {name:<18}: {val:.4f}  ({diff:+.2f}% vs Full)")

    # 8. 统计检验
    logger.info("\n" + "=" * 70)
    logger.info("统计显著性检验")
    logger.info("=" * 70)

    ensemble_scores = compute_per_query_ndcg(results_ensemble, qrels)
    stat_results = []

    for method_name, method_results in [
        ('RRF', results_rrf),
        ('Linear', results_linear),
        ('BM25', results_bm25),
        ('Dense-Small', results_dense_s),
    ]:
        baseline_scores = compute_per_query_ndcg(method_results, qrels)
        stat = statistical_tests(ensemble_scores, baseline_scores, 'MoRE-Ensemble', method_name)
        stat_results.append(stat)

        sig = '***' if stat['p_value_t'] < 0.001 else '**' if stat['p_value_t'] < 0.01 else '*' if stat['p_value_t'] < 0.05 else 'n.s.'
        logger.info(f"  vs {method_name:<14}: p={stat['p_value_t']:.2e} {sig}  d={stat['cohens_d']:.3f}  CI=[{stat['ci_95'][0]:.4f}, {stat['ci_95'][1]:.4f}]")

    # 9. 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'name': 'Private Industrial PDF Documents',
            'n_documents': 10,
            'n_chunks': len(corpus),
            'n_queries': len(queries),
        },
        'methods': all_metrics,
        'weights': {
            'MoRE-Bayes': {
                'BM25': weights_2[0],
                'Dense-Small': weights_2[1],
            },
            'MoRE-Ensemble': {
                'BM25': weights_3[0],
                'Dense-Small': weights_3[1],
                'Dense-Large': weights_3[2],
            },
        },
        'expert_stats': {
            'Bayes_2expert': [
                {'rho': s['rho'], 'sigma2': s['sigma2']} for s in stats_2
            ],
            'Ensemble_3expert': [
                {'rho': s['rho'], 'sigma2': s['sigma2']} for s in stats_3
            ],
        },
        'ablation': ablation,
        'statistical_tests': stat_results,
        'latency': {
            'BM25_index_s': bm25_index_time,
            'Dense_Small_index_s': dense_s_time,
            'Dense_Large_index_s': dense_l_time,
            'BM25_search_ms': bm25_latency,
            'Dense_Small_search_ms': dense_s_latency,
            'Dense_Large_search_ms': dense_l_latency,
        },
    }

    output_file = output_dir / f'private_pdf_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果已保存到: {output_file}")
    logger.info("实验完成!")


if __name__ == '__main__':
    main()
