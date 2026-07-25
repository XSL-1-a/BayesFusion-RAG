#!/usr/bin/env python3
"""
准备真实 MaintNet 数据用于 MoRE 实验
将 CSV 维护记录转换为检索实验所需的格式

数据来源: data/maintnet/{aviation,automotive,facility}.csv
格式: IDENT, PROBLEM, ACTION

转换逻辑:
- PROBLEM 作为查询
- ACTION 作为相关文档
- 其他记录的 ACTION 作为负例候选

作者: RAG Research Team
日期: 2026-01-28
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_maintnet_csv(csv_path: str, domain: str) -> List[Dict]:
    """加载 MaintNet CSV 文件 (支持不同格式)"""
    records = []

    # 不同领域的列映射
    column_mappings = {
        'aviation': {'id': 'IDENT', 'problem': 'PROBLEM', 'action': 'ACTION'},
        'automotive': {'id': 'jobno', 'problem': 'Reason', 'action': 'Notes'},
        'facility': {'id': 'WORK_ID', 'problem': 'PROB_TYPE', 'action': 'DESCRIPTION'}
    }

    mapping = column_mappings.get(domain, column_mappings['aviation'])

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_col = mapping['problem']
            action_col = mapping['action']
            id_col = mapping['id']

            problem = row.get(problem_col, '').strip()
            action = row.get(action_col, '').strip()

            # 跳过空记录
            if not problem or not action:
                continue

            # 过滤过短的记录
            if len(problem) < 3 or len(action) < 5:
                continue

            records.append({
                'id': row.get(id_col, str(len(records))),
                'problem': problem,
                'action': action
            })
    return records


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """从文本中提取实体 (简单版本)"""
    text_lower = text.lower()

    components = {
        'engine', 'propeller', 'magneto', 'carburetor', 'cylinder', 'piston',
        'valve', 'gasket', 'fuel pump', 'oil pump', 'throttle', 'alternator',
        'battery', 'fuel line', 'oil filter', 'brake', 'tire', 'wheel',
        'hose', 'seal', 'bearing', 'transmission', 'radiator', 'hvac',
        'compressor', 'pump', 'motor', 'pipe', 'drain', 'faucet',
        'rocker', 'cover', 'baffle', 'intake', 'exhaust'
    }

    symptoms = {
        'leaking', 'cracked', 'worn', 'broken', 'failed', 'noisy', 'loose',
        'corroded', 'overheating', 'damaged', 'missing', 'stuck', 'seized',
        'rough', 'idle', 'fouled'
    }

    actions = {
        'replace', 'replaced', 'repair', 'install', 'remove', 'inspect',
        'adjust', 'clean', 'tighten', 'test', 'check', 'calibrate',
        'removed', 'installed', 'adjusted', 'cleaned', 'tightened'
    }

    found_components = [c for c in components if c in text_lower]
    found_symptoms = [s for s in symptoms if s in text_lower]
    found_actions = [a for a in actions if a in text_lower]

    return {
        'components': found_components,
        'symptoms': found_symptoms,
        'actions': found_actions
    }


def compute_entity_overlap(query_entities: Dict, doc_entities: Dict) -> float:
    """计算实体重叠分数"""
    score = 0.0

    # 组件匹配 (高权重)
    q_comps = set(query_entities.get('components', []))
    d_comps = set(doc_entities.get('components', []))
    comp_overlap = len(q_comps & d_comps)
    score += comp_overlap * 3.0

    # 症状匹配
    q_syms = set(query_entities.get('symptoms', []))
    d_syms = set(doc_entities.get('symptoms', []))
    sym_overlap = len(q_syms & d_syms)
    score += sym_overlap * 2.0

    # 动作匹配
    q_acts = set(query_entities.get('actions', []))
    d_acts = set(doc_entities.get('actions', []))
    act_overlap = len(q_acts & d_acts)
    score += act_overlap * 1.0

    return score


def compute_bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75,
                       avg_dl: float = 50.0) -> float:
    """计算 BM25 分数"""
    q_tokens = query.lower().split()
    d_tokens = doc.lower().split()
    doc_len = len(d_tokens)

    # 简化版本: 不使用IDF (单文档)
    score = 0.0
    for qt in q_tokens:
        tf = d_tokens.count(qt)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
            score += numerator / (denominator + 1e-10)

    return score


def create_retrieval_dataset(
    records: List[Dict],
    domain: str,
    n_queries: int = 200,
    n_candidates: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    创建检索评估数据集

    策略:
    - 对于每个查询 (PROBLEM)，其对应的 ACTION 是相关文档
    - 从其他记录中采样作为负例候选
    - 计算 BM25 和 Entity 分数
    """
    np.random.seed(seed)

    # 限制查询数量
    if len(records) > n_queries:
        query_indices = np.random.choice(len(records), n_queries, replace=False)
    else:
        query_indices = list(range(len(records)))

    # 预提取所有记录的实体
    all_entities = {}
    for i, rec in enumerate(records):
        all_entities[i] = {
            'problem': extract_entities_from_text(rec['problem']),
            'action': extract_entities_from_text(rec['action'])
        }

    # 计算平均文档长度
    avg_doc_len = np.mean([len(rec['action'].split()) for rec in records])

    queries = []
    candidates_dict = {}
    ground_truth = {}
    expert_scores = {}  # {query_id: {expert: [scores]}}

    for q_idx in query_indices:
        rec = records[q_idx]
        query_id = f"{domain}_q{len(queries)+1}"
        query_text = rec['problem']

        queries.append({
            'query_id': query_id,
            'query': query_text,
            'original_id': rec['id']
        })

        # 构建候选池
        # 正例: 当前记录的 ACTION
        # 负例: 随机采样其他记录的 ACTION

        candidate_indices = [q_idx]  # 正例

        # 采样负例
        other_indices = [i for i in range(len(records)) if i != q_idx]
        if len(other_indices) > n_candidates - 1:
            neg_indices = np.random.choice(other_indices, n_candidates - 1, replace=False)
        else:
            neg_indices = other_indices
        candidate_indices.extend(neg_indices)

        # 打乱顺序
        np.random.shuffle(candidate_indices)

        # 构建候选文档
        candidates = []
        bm25_scores = []
        entity_scores = []
        dense_scores = []
        image_scores = []
        relevance = []

        query_entities = all_entities[q_idx]['problem']

        for c_idx in candidate_indices:
            c_rec = records[c_idx]
            doc_id = f"{domain}_doc_{c_idx}"
            doc_content = c_rec['action']

            # 计算 BM25 分数
            bm25 = compute_bm25_score(query_text, doc_content, avg_dl=avg_doc_len)
            bm25_scores.append(bm25)

            # 计算 Entity 分数
            doc_entities = all_entities[c_idx]['action']
            entity = compute_entity_overlap(query_entities, doc_entities)
            entity_scores.append(entity)

            # Dense 分数 (模拟: 基于词重叠 + 噪声)
            q_words = set(query_text.lower().split())
            d_words = set(doc_content.lower().split())
            overlap = len(q_words & d_words) / (len(q_words) + 1e-10)
            dense = overlap + np.random.normal(0, 0.1)
            dense_scores.append(max(0, dense))

            # Image 分数 (模拟: 基于视觉关键词)
            visual_keywords = {'diagram', 'figure', 'photo', 'image', 'view', 'picture'}
            visual_count = sum(1 for kw in visual_keywords if kw in doc_content.lower())
            image = visual_count * 0.2 + np.random.normal(0, 0.05)
            image_scores.append(max(0, image))

            # 相关性标签
            is_relevant = 1.0 if c_idx == q_idx else 0.0
            relevance.append(is_relevant)

            candidates.append({
                'doc_id': doc_id,
                'content': doc_content,
                'original_id': c_rec['id']
            })

        # 归一化分数
        def normalize(scores):
            scores = np.array(scores)
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s > 1e-6:
                return ((scores - min_s) / (max_s - min_s)).tolist()
            return [0.5] * len(scores)

        candidates_dict[query_id] = candidates
        ground_truth[query_id] = [
            candidates[i]['doc_id']
            for i, r in enumerate(relevance) if r == 1.0
        ]

        expert_scores[query_id] = {
            'BM25': normalize(bm25_scores),
            'Dense': normalize(dense_scores),
            'Entity': normalize(entity_scores),
            'Image': normalize(image_scores),
            'relevance': relevance
        }

    return {
        'queries': queries,
        'candidates': candidates_dict,
        'ground_truth': ground_truth,
        'expert_scores': expert_scores,
        'metadata': {
            'domain': domain,
            'n_queries': len(queries),
            'n_candidates': n_candidates,
            'n_total_records': len(records),
            'created_at': datetime.now().isoformat()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare real MaintNet data for MoRE')
    parser.add_argument('--data_dir', type=str, default='./data/maintnet',
                        help='MaintNet CSV directory')
    parser.add_argument('--output_dir', type=str,
                        default='./data/real_experiments',
                        help='Output directory')
    parser.add_argument('--n_queries', type=int, default=200,
                        help='Number of queries per domain')
    parser.add_argument('--n_candidates', type=int, default=50,
                        help='Number of candidates per query')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Preparing Real MaintNet Data for MoRE Experiments")
    logger.info("=" * 60)

    domains = ['aviation', 'automotive', 'facility']
    all_stats = {}

    for domain in domains:
        csv_path = data_dir / f'{domain}.csv'
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            continue

        logger.info(f"\nProcessing {domain.upper()}...")

        # 加载数据
        records = load_maintnet_csv(str(csv_path), domain)
        logger.info(f"  Loaded {len(records)} records from {csv_path}")

        # 创建检索数据集
        n_queries = min(args.n_queries, len(records))
        dataset = create_retrieval_dataset(
            records, domain,
            n_queries=n_queries,
            n_candidates=args.n_candidates,
            seed=args.seed
        )

        # 保存
        domain_dir = output_dir / domain
        domain_dir.mkdir(exist_ok=True)

        with open(domain_dir / 'queries.json', 'w', encoding='utf-8') as f:
            json.dump(dataset['queries'], f, indent=2, ensure_ascii=False)

        with open(domain_dir / 'candidates.json', 'w', encoding='utf-8') as f:
            json.dump(dataset['candidates'], f, indent=2, ensure_ascii=False)

        with open(domain_dir / 'ground_truth.json', 'w', encoding='utf-8') as f:
            json.dump(dataset['ground_truth'], f, indent=2, ensure_ascii=False)

        with open(domain_dir / 'expert_scores.json', 'w', encoding='utf-8') as f:
            json.dump(dataset['expert_scores'], f, indent=2, ensure_ascii=False)

        logger.info(f"  Created {len(dataset['queries'])} queries")
        logger.info(f"  Saved to {domain_dir}")

        all_stats[domain] = dataset['metadata']

    # 保存统计
    with open(output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("Data Preparation Complete")
    logger.info("=" * 60)

    for domain, stats in all_stats.items():
        logger.info(f"{domain}: {stats['n_queries']} queries, {stats['n_total_records']} total records")


if __name__ == '__main__':
    main()
