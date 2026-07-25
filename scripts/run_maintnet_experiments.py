#!/usr/bin/env python3
"""
MaintNet 数据集实验脚本 v4
改进: (1) 使用全部测试数据 (2) HMR v2: BM25 doc-level + Dense chunk-level
      (3) 实体置信度门控 (4) 改进查询生成
      (5) 新增 RRF 和 HyDE 强基线
"""

import os
import sys
import json
import csv
import re
import math
import random
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.utils.embedder import HuggingFaceEmbedder
from src.utils.config import EmbeddingConfig
from src.retrieval.indexer import VectorIndex, IndexItem

# ============= 配置 =============
MAINTNET_DIR = project_root / "data" / "maintnet"
OUTPUT_DIR = project_root / "results" / "maintnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_BGE_PATH = str(project_root / "models" / "bge-small-zh-v1.5")
LOCAL_QWEN_PATH = str(project_root / "models" / "Qwen2-1.5B-Instruct")

TOP_K = 5
RANDOM_SEED = 42
TRAIN_RATIO = 0.8   # 知识库占比
NUM_QUERIES = 0      # 0 = 使用全部测试记录 (最大统计能力)
RRF_K = 60           # RRF 常数 (标准值)


def get_local_embedder():
    config = EmbeddingConfig(provider="huggingface", model=LOCAL_BGE_PATH, dimension=512)
    return HuggingFaceEmbedder(config)


# =====================================================================
# LLM 加载 (用于 HyDE)
# =====================================================================

def load_qwen_llm(device: str = "auto"):
    """加载本地 Qwen2-1.5B-Instruct 用于 HyDE 生成"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载 Qwen2-1.5B-Instruct ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_QWEN_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_QWEN_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_hypothetical_doc(query: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    """用 LLM 生成假设性维修文档 (HyDE)"""
    import torch

    messages = [
        {"role": "system", "content": "You are an experienced maintenance technician. "
         "Given a maintenance query, write a brief, realistic maintenance record "
         "that would answer the query. Include the problem observed and the corrective action taken. "
         "Be concise and technical."},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def batch_generate_hyde_docs(
    qa_pairs_by_domain: Dict[str, List],
    model, tokenizer
) -> Dict[str, List[str]]:
    """为所有查询批量生成 HyDE 假设文档"""
    hyde_docs = {}
    total = sum(len(qas) for qas in qa_pairs_by_domain.values())
    done = 0

    for domain, qa_pairs in qa_pairs_by_domain.items():
        docs = []
        for i, qa in enumerate(qa_pairs):
            doc = generate_hypothetical_doc(qa.query, model, tokenizer)
            docs.append(doc)
            done += 1
            if done % 50 == 0:
                print(f"    HyDE 生成进度: {done}/{total}", flush=True)
        hyde_docs[domain] = docs

    return hyde_docs


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class MaintenanceRecord:
    record_id: str
    domain: str
    problem: str
    action: str
    category: str = ""          # 问题类别 (用于 facility)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    query_id: str
    domain: str
    query: str                  # 自然语言查询 (不含原文)
    answer: str                 # 地面真值答案
    source_record_id: str       # 来源记录 (不在知识库中)
    relevant_kb_ids: List[str]  # 知识库中的相关记录 ID


# =====================================================================
# 1. 数据加载与切分 (消除数据泄露)
# =====================================================================

def load_and_split_data() -> Dict[str, Tuple[List[MaintenanceRecord], List[MaintenanceRecord]]]:
    """加载数据并切分为知识库 / 测试集 (80/20)"""
    print("\n[1/7] 加载并切分数据...")
    random.seed(RANDOM_SEED)
    result = {}

    # --- Aviation ---
    path = MAINTNET_DIR / "aviation.csv"
    if path.exists():
        records = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for i, row in enumerate(csv.DictReader(f)):
                p = row.get('PROBLEM', '').strip()
                a = row.get('ACTION', '').strip()
                if p and a and len(p) > 10 and len(a) > 10:
                    records.append(MaintenanceRecord(
                        record_id=f"avi_{i}", domain="aviation",
                        problem=p, action=a,
                        metadata={"ident": row.get('IDENT', '')}
                    ))
        random.shuffle(records)
        split = int(len(records) * TRAIN_RATIO)
        result['aviation'] = (records[:split], records[split:])
        print(f"  Aviation: KB={split}, Test={len(records)-split}")

    # --- Automotive ---
    path = MAINTNET_DIR / "automotive.csv"
    if path.exists():
        records = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for i, row in enumerate(csv.DictReader(f)):
                reason = row.get('Reason', '').strip()
                notes = row.get('Notes', '').strip()
                if reason and notes and len(notes) > 10:
                    records.append(MaintenanceRecord(
                        record_id=f"auto_{i}", domain="automotive",
                        problem=reason, action=notes,
                        category=reason,
                        metadata={"dept": row.get('Dept', '')}
                    ))
        random.shuffle(records)
        split = int(len(records) * TRAIN_RATIO)
        result['automotive'] = (records[:split], records[split:])
        print(f"  Automotive: KB={split}, Test={len(records)-split}")

    # --- Facility ---
    path = MAINTNET_DIR / "facility.csv"
    if path.exists():
        records = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for i, row in enumerate(csv.DictReader(f)):
                pt = row.get('PROB_TYPE', '').strip()
                desc = row.get('DESCRIPTION', '').strip()
                if pt and desc and len(desc) > 10:
                    records.append(MaintenanceRecord(
                        record_id=f"fac_{i}", domain="facility",
                        problem=pt, action=desc,
                        category=pt,
                        metadata={"site_id": row.get('SITE_ID', '')}
                    ))
        random.shuffle(records)
        split = int(len(records) * TRAIN_RATIO)
        result['facility'] = (records[:split], records[split:])
        print(f"  Facility: KB={split}, Test={len(records)-split}")

    return result


# =====================================================================
# 2. 文本处理和 BM25 评分器
# =====================================================================

_QUERY_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
    'through', 'during', 'before', 'after', 'and', 'but', 'or', 'not',
    'no', 'as', 'if', 'it', 'its', 'this', 'that', 'these', 'those',
    'such', 'than', 'what', 'how', 'which', 'when', 'where', 'who',
    'why', 'there', 'here', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'any', 'new', 'used', 'also',
    'between', 'while', 'another', 'spare',
}


def _normalize(text: str) -> str:
    """标准化文本用于匹配"""
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).strip()

def _token_set(text: str) -> Set[str]:
    return set(_normalize(text).split())

def _jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


class BM25Scorer:
    """
    预计算 BM25 评分器
    预计算文档侧的 TF、IDF、文档长度等，加速逐查询评分
    """

    def __init__(self, records: List[MaintenanceRecord], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1 = k1
        self.b = b
        self.N = len(records)

        # 预计算文档侧信息
        self.doc_texts = [_normalize(f"{r.problem} {r.action}") for r in records]
        self.df: Counter = Counter()
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []

        for dt in self.doc_texts:
            terms = dt.split()
            self.doc_lens.append(len(terms))
            tf = Counter(terms)
            self.doc_tfs.append(tf)
            for t in set(terms):
                self.df[t] += 1

        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0

    def score_query(self, query: str) -> Dict[str, float]:
        """返回 {record_id: bm25_score} 字典"""
        query_terms = _normalize(query).split()
        scores = {}
        for i, rec in enumerate(self.records):
            score = 0.0
            dl = self.doc_lens[i]
            tf_map = self.doc_tfs[i]
            for qt in query_terms:
                if qt in tf_map:
                    tf = tf_map[qt]
                    df_val = self.df.get(qt, 0)
                    idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
                    tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    score += idf * tf_norm
            scores[rec.record_id] = score
        return scores

    def score_query_top_k(self, query: str, top_k: int) -> List[Tuple[MaintenanceRecord, float]]:
        """返回 top-k (record, score) 列表"""
        scores = self.score_query(query)
        scored = [(self.records[i], scores[self.records[i].record_id]) for i in range(len(self.records))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# =====================================================================
# 3. 查询生成 (无数据泄露)
# =====================================================================

AVIATION_TEMPLATES = [
    "What maintenance action is recommended when a {component} shows signs of {symptom}?",
    "How should a technician address {symptom} in the {component} system?",
    "What is the standard repair procedure for {component} {symptom}?",
    "Describe the corrective action for {component} failure involving {symptom}.",
    "What steps should be taken when {symptom} is detected in the {component}?",
]

FACILITY_TEMPLATES = [
    "What maintenance is needed for a {category} issue at a facility?",
    "How should facility staff handle a {category} problem?",
    "What repair work is typically done for {category} maintenance?",
    "Describe the service procedure for {category} related issues.",
    "What corrective actions are taken for {category} problems in buildings?",
]

GENERIC_TEMPLATES = [
    "What is the recommended maintenance action for {problem_abstract}?",
    "How should the issue of {problem_abstract} be resolved?",
    "What repair procedure addresses {problem_abstract}?",
    "Describe the maintenance steps for handling {problem_abstract}.",
]


def _extract_component_symptom(problem: str) -> Tuple[str, str]:
    """从问题描述中提取组件和症状 (改进版: 更好的回退策略)"""
    p_lower = problem.lower()

    # 常见组件 (按长度降序匹配，优先匹配更具体的)
    components = sorted([
        'intake gasket', 'rocker cover', 'exhaust gasket', 'oil cooler',
        'spark plug', 'fuel servo', 'fuel line', 'fuel tank', 'oil filter',
        'air filter', 'fuel pump', 'oil pump', 'vacuum pump', 'trim tab',
        'pitot tube', 'nose gear', 'landing gear', 'cv joint', 'ball joint',
        'tie rod', 'wheel bearing', 'master cylinder', 'oxygen sensor',
        'fuel filter', 'cabin filter', 'shock absorber', 'catalytic converter',
        'water pump', 'water heater', 'air conditioner', 'fire alarm',
        'smoke detector', 'electrical panel', 'circuit breaker',
        'exhaust', 'engine', 'cylinder', 'piston', 'crankshaft', 'camshaft',
        'magneto', 'carburetor', 'propeller', 'alternator', 'starter',
        'battery', 'throttle', 'muffler', 'valve', 'gasket', 'baffle',
        'cowling', 'hose', 'brake', 'tire', 'wheel', 'strut',
        'aileron', 'elevator', 'rudder', 'flap', 'wing',
        'firewall', 'windshield', 'altimeter', 'tachometer', 'gyro',
        'seal', 'bearing', 'bushing', 'bolt', 'primer', 'turbocharger',
        'transmission', 'radiator', 'thermostat', 'ignition', 'distributor',
        'coil', 'steering', 'suspension', 'spring', 'axle', 'caliper',
        'rotor', 'drum', 'pad', 'headlight', 'wiper',
        'hvac', 'boiler', 'furnace', 'compressor', 'ductwork', 'blower',
        'fan', 'motor', 'pump', 'pipe', 'drain', 'faucet', 'toilet', 'sink',
        'sprinkler', 'outlet', 'switch', 'light', 'fixture', 'ballast',
        'transformer', 'generator', 'door', 'lock', 'window', 'roof',
    ], key=len, reverse=True)

    symptoms = [
        'leaking', 'leaked', 'leak', 'cracked', 'crack', 'worn', 'broken',
        'failed', 'failure', 'noisy', 'noise', 'rough', 'loose',
        'corroded', 'corrosion', 'overheating', 'overheat', 'vibration',
        'fouled', 'damaged', 'damage', 'missing', 'stuck', 'seized',
        'intermittent', 'inoperative', 'inop', 'excessive',
        'low compression', 'loss of power', 'hard starting', 'rough idle',
        'oil consumption',
    ]

    # 组件匹配
    found_comp = None
    for c in components:
        if c in p_lower:
            found_comp = c
            break

    # 回退: 提取长词作为组件候选
    if found_comp is None:
        words = re.findall(r'[a-z]{5,}', p_lower)
        for w in words:
            if w not in _QUERY_STOP_WORDS and w not in {
                'removed', 'replaced', 'installed', 'inspected', 'found',
                'noted', 'reported', 'required', 'performed', 'checked',
                'aircraft', 'maintenance', 'annual', 'condition',
            }:
                found_comp = w
                break
    found_comp = found_comp or 'equipment'

    # 症状匹配
    found_symp = None
    for s in sorted(symptoms, key=len, reverse=True):
        if s in p_lower:
            found_symp = s
            break

    # 回退: 检查常见故障模式
    if found_symp is None:
        if any(w in p_lower for w in ['not work', 'won\'t', 'unable', 'inop']):
            found_symp = 'failure'
        elif any(w in p_lower for w in ['wear', 'worn', 'deteriorat']):
            found_symp = 'wear'
        elif any(w in p_lower for w in ['replace', 'chang', 'due for']):
            found_symp = 'deterioration'
    found_symp = found_symp or 'malfunction'

    return found_comp, found_symp


def generate_qa_pairs(
    data: Dict[str, Tuple[List[MaintenanceRecord], List[MaintenanceRecord]]]
) -> Dict[str, Tuple[List[MaintenanceRecord], List[QAPair]]]:
    """
    从测试集生成 QA 对，并找到知识库中的相关记录
    查询使用抽象化模板，不直接包含 problem 原文
    使用全部测试数据 (不做子采样)
    """
    print("\n[2/7] 生成 QA 数据集 (无数据泄露)...")
    random.seed(RANDOM_SEED)
    result = {}

    for domain, (kb_records, test_records) in data.items():
        qa_pairs = []
        # 预计算 KB 的 token 集合
        kb_problem_tokens = [(r, _token_set(r.problem)) for r in kb_records]
        kb_action_tokens = [(r, _token_set(r.action)) for r in kb_records]

        # 使用全部测试数据或指定数量
        if NUM_QUERIES > 0 and NUM_QUERIES < len(test_records):
            sampled = random.sample(test_records, NUM_QUERIES)
        else:
            sampled = list(test_records)  # 全部使用

        for i, record in enumerate(sampled):
            # 生成抽象化查询
            if domain == 'aviation':
                comp, symp = _extract_component_symptom(record.problem)
                template = random.choice(AVIATION_TEMPLATES)
                query = template.format(component=comp, symptom=symp)
            elif domain == 'facility':
                template = random.choice(FACILITY_TEMPLATES)
                query = template.format(category=record.category.lower())
            else:
                # 抽象化 problem: 取关键词
                words = _normalize(record.problem).split()
                abstract = ' '.join(w for w in words if len(w) > 3)[:80]
                template = random.choice(GENERIC_TEMPLATES)
                query = template.format(problem_abstract=abstract or record.problem[:50])

            # 找 KB 中的相关记录
            test_prob_tokens = _token_set(record.problem)
            test_act_tokens = _token_set(record.action)
            relevant_ids = []
            for j, (kb_rec, kb_prob_tok) in enumerate(kb_problem_tokens):
                prob_sim = _jaccard(test_prob_tokens, kb_prob_tok)
                if prob_sim > 0.5:
                    relevant_ids.append(kb_rec.record_id)
                elif prob_sim > 0.3:
                    _, kb_act_tok = kb_action_tokens[j]
                    act_sim = _jaccard(test_act_tokens, kb_act_tok)
                    if act_sim > 0.4:
                        relevant_ids.append(kb_rec.record_id)

            qa_pairs.append(QAPair(
                query_id=f"{domain}_q{i}",
                domain=domain,
                query=query,
                answer=record.action,
                source_record_id=record.record_id,
                relevant_kb_ids=relevant_ids
            ))

        result[domain] = (kb_records, qa_pairs)
        n_with_rel = sum(1 for qa in qa_pairs if qa.relevant_kb_ids)
        avg_rel = np.mean([len(qa.relevant_kb_ids) for qa in qa_pairs])
        print(f"  {domain}: {len(qa_pairs)} QA对, {n_with_rel} 个有KB相关记录 "
              f"(avg {avg_rel:.1f} 相关文档)")

    return result


# =====================================================================
# 4. 实体模块 (IDF 加权 + 置信度门控)
# =====================================================================

class EntityKnowledgeBase:
    """
    增强版实体知识库 v2
    - IDF 加权: 稀有实体匹配更有价值
    - 组件-症状协同: 同时匹配组件+症状给更高分
    - 置信度门控: 实体信号不可靠时自动关闭 (防止噪声)
    """

    COMPONENTS = {
        # 航空
        'engine', 'propeller', 'magneto', 'carburetor', 'cylinder', 'piston',
        'crankshaft', 'camshaft', 'valve', 'spark plug', 'fuel pump', 'oil pump',
        'fuel servo', 'throttle', 'alternator', 'starter', 'battery',
        'fuel line', 'fuel tank', 'oil filter', 'air filter', 'exhaust', 'muffler',
        'landing gear', 'brake', 'tire', 'wheel', 'strut', 'nose gear',
        'aileron', 'elevator', 'rudder', 'flap', 'trim tab', 'wing',
        'cowling', 'firewall', 'windshield', 'pitot tube',
        'altimeter', 'tachometer', 'vacuum pump', 'gyro',
        'hose', 'gasket', 'seal', 'bearing', 'bushing', 'bolt',
        'intake gasket', 'rocker cover', 'exhaust gasket', 'oil cooler',
        'primer', 'baffle', 'turbocharger', 'wastegate',
        # 汽车
        'transmission', 'radiator', 'thermostat', 'water pump',
        'ignition', 'distributor', 'coil', 'catalytic converter',
        'oxygen sensor', 'fuel filter', 'cabin filter',
        'steering', 'suspension', 'shock absorber', 'spring',
        'axle', 'cv joint', 'ball joint', 'tie rod', 'wheel bearing',
        'caliper', 'rotor', 'drum', 'pad', 'master cylinder',
        'headlight', 'taillight', 'wiper',
        # 设施
        'hvac', 'boiler', 'furnace', 'air conditioner', 'compressor',
        'ductwork', 'blower', 'fan', 'motor', 'pump',
        'pipe', 'drain', 'faucet', 'toilet', 'sink',
        'water heater', 'sprinkler', 'fire alarm', 'smoke detector',
        'electrical panel', 'circuit breaker', 'outlet', 'switch',
        'light', 'fixture', 'ballast', 'transformer', 'generator',
        'door', 'lock', 'window', 'roof',
        'floor', 'wall', 'ceiling', 'carpet', 'tile',
    }

    ACTIONS = {
        'replace', 'replaced', 'repair', 'repaired', 'remove', 'removed',
        'install', 'installed', 'inspect', 'inspected', 'adjust', 'adjusted',
        'clean', 'cleaned', 'lubricate', 'lubricated', 'tighten', 'tightened',
        'torque', 'torqued', 'test', 'tested', 'check', 'checked',
        'calibrate', 'overhaul', 'overhauled', 'rebuild', 'rebuilt',
        'reseal', 'resealed', 'resurface', 'drain', 'drained', 'flush',
        'flushed', 'bleed', 'service', 'serviced', 'align', 'aligned',
    }

    SYMPTOMS = {
        'leaking', 'leaked', 'leak', 'cracked', 'crack', 'worn', 'broken',
        'failed', 'failure', 'noisy', 'noise', 'rough', 'loose',
        'corroded', 'corrosion', 'overheating', 'overheat', 'vibration',
        'fouled', 'damaged', 'damage', 'missing', 'stuck', 'seized',
        'intermittent', 'inoperative', 'inop', 'low', 'high', 'excessive',
    }

    def __init__(self):
        self._doc_components: Dict[str, Set[str]] = defaultdict(set)
        self._doc_actions: Dict[str, Set[str]] = defaultdict(set)
        self._doc_symptoms: Dict[str, Set[str]] = defaultdict(set)
        self._entity_doc_count: Counter = Counter()  # 计算 IDF
        self._total_docs = 0

    def index_document(self, doc_id: str, text: str):
        text_lower = text.lower()
        self._total_docs += 1

        found_entities = set()
        for comp in self.COMPONENTS:
            if comp in text_lower:
                self._doc_components[doc_id].add(comp)
                found_entities.add(comp)

        for act in self.ACTIONS:
            if f' {act} ' in f' {text_lower} ' or text_lower.startswith(act) or text_lower.endswith(act):
                self._doc_actions[doc_id].add(act)
                found_entities.add(act)

        for sym in self.SYMPTOMS:
            if sym in text_lower:
                self._doc_symptoms[doc_id].add(sym)
                found_entities.add(sym)

        for e in found_entities:
            self._entity_doc_count[e] += 1

    def get_idf(self, entity: str) -> float:
        df = self._entity_doc_count.get(entity, 0)
        if df == 0:
            return 0.0
        return math.log(self._total_docs / df)

    def get_doc_entities(self, doc_id: str) -> Tuple[Set[str], Set[str], Set[str]]:
        return (self._doc_components.get(doc_id, set()),
                self._doc_actions.get(doc_id, set()),
                self._doc_symptoms.get(doc_id, set()))

    def extract_from_query(self, query: str) -> Tuple[Set[str], Set[str], Set[str]]:
        q = query.lower()
        comps = {c for c in self.COMPONENTS if c in q}
        acts = {a for a in self.ACTIONS if a in q}
        syms = {s for s in self.SYMPTOMS if s in q}
        return comps, acts, syms

    def compute_entity_score(
        self,
        query_comps: Set[str], query_acts: Set[str], query_syms: Set[str],
        doc_id: str
    ) -> float:
        """IDF加权实体匹配分数 + 组件-症状协同"""
        doc_comps, doc_acts, doc_syms = self.get_doc_entities(doc_id)

        score = 0.0

        # 组件匹配 (IDF 加权)
        comp_matches = query_comps & doc_comps
        for c in comp_matches:
            score += self.get_idf(c) * 1.0

        # 症状匹配 (IDF 加权)
        sym_matches = query_syms & doc_syms
        for s in sym_matches:
            score += self.get_idf(s) * 0.8

        # 动作匹配
        act_matches = query_acts & doc_acts
        for a in act_matches:
            score += self.get_idf(a) * 0.5

        # 协同加成: 同时匹配组件 + 症状 → 2x
        if comp_matches and sym_matches:
            score *= 2.0

        return score

    @property
    def total_entities(self) -> int:
        return sum(len(v) for v in self._doc_components.values()) + \
               sum(len(v) for v in self._doc_actions.values()) + \
               sum(len(v) for v in self._doc_symptoms.values())


# =====================================================================
# 5. 索引构建
# =====================================================================

def build_chunk_indices(
    domain_kb: Dict[str, List[MaintenanceRecord]],
    embedder
) -> Dict[str, VectorIndex]:
    """为知识库构建 chunk 向量索引"""
    print("\n[3/7] 构建向量索引 (仅知识库)...")
    indices = {}

    for domain, records in domain_kb.items():
        print(f"  {domain}: {len(records)} 条记录...", end=" ", flush=True)

        chunk_index = VectorIndex(embedder.dimension)
        chunk_texts = [f"Problem: {r.problem}\nAction: {r.action}" for r in records]
        chunk_embs = embedder.embed_texts(chunk_texts)
        for r, emb in zip(records, chunk_embs):
            chunk_index.add(IndexItem(
                item_id=f"{r.record_id}_chunk",
                content=f"Problem: {r.problem}\nAction: {r.action}",
                embedding=emb,
                metadata={"domain": domain, "doc_id": r.record_id,
                          "problem": r.problem, "action": r.action}
            ))
        chunk_index.build()

        indices[domain] = chunk_index
        print(f"Chunks={len(chunk_index.items)}")

    return indices


def build_bm25_scorers(
    domain_kb: Dict[str, List[MaintenanceRecord]]
) -> Dict[str, BM25Scorer]:
    """构建预计算 BM25 评分器"""
    print("\n[3.5/7] 构建 BM25 评分器...")
    scorers = {}
    for domain, records in domain_kb.items():
        scorers[domain] = BM25Scorer(records)
        print(f"  {domain}: {len(records)} 文档, avgdl={scorers[domain].avgdl:.1f}")
    return scorers


def build_entity_stores(
    domain_kb: Dict[str, List[MaintenanceRecord]]
) -> Dict[str, EntityKnowledgeBase]:
    print("\n[4/7] 构建实体知识库 (IDF加权)...")
    stores = {}
    for domain, records in domain_kb.items():
        ekb = EntityKnowledgeBase()
        for r in records:
            ekb.index_document(r.record_id, f"{r.problem} {r.action}")
        stores[domain] = ekb
        print(f"  {domain}: {ekb.total_entities} 实体实例, "
              f"{len(ekb._entity_doc_count)} 唯一实体")
    return stores


# =====================================================================
# 6. 检索方法
# =====================================================================

def retrieve_bm25(
    query: str,
    bm25_scorer: BM25Scorer,
    records: List[MaintenanceRecord],
    top_k: int = TOP_K
) -> List[Dict]:
    """BM25 检索 (使用预计算评分器)"""
    scored = bm25_scorer.score_query_top_k(query, top_k)
    return [
        {"content": f"Problem: {r.problem}\nAction: {r.action}",
         "score": s, "doc_id": r.record_id,
         "problem": r.problem, "action": r.action}
        for r, s in scored
    ]


def retrieve_dense(
    query: str, embedder, chunk_index: VectorIndex, top_k: int = TOP_K
) -> List[Dict]:
    """Dense 向量检索"""
    q_emb = embedder.embed_text(query)
    results = chunk_index.search(q_emb, k=top_k)
    return [
        {"content": item.content, "score": score,
         "doc_id": item.metadata.get("doc_id", ""),
         "problem": item.metadata.get("problem", ""),
         "action": item.metadata.get("action", "")}
        for item, score in results
    ]


def retrieve_hybrid(
    query: str, embedder,
    bm25_scorer: BM25Scorer,
    records: List[MaintenanceRecord],
    chunk_index: VectorIndex,
    alpha: float = 0.5,
    top_k: int = TOP_K
) -> List[Dict]:
    """Hybrid: BM25 + Dense 线性融合 (同层级)"""
    bm25_results = retrieve_bm25(query, bm25_scorer, records, top_k=top_k * 3)
    dense_results = retrieve_dense(query, embedder, chunk_index, top_k=top_k * 3)

    bm25_scores = {r['doc_id']: r for r in bm25_results}
    dense_scores = {r['doc_id']: r for r in dense_results}
    all_ids = set(bm25_scores) | set(dense_scores)

    bm25_max = max((r['score'] for r in bm25_results), default=1.0) or 1.0
    dense_max = max((r['score'] for r in dense_results), default=1.0) or 1.0

    fused = []
    for did in all_ids:
        bm25_s = bm25_scores.get(did, {}).get('score', 0) / bm25_max
        dense_s = dense_scores.get(did, {}).get('score', 0) / dense_max
        combined = alpha * bm25_s + (1 - alpha) * dense_s
        ref = bm25_scores.get(did) or dense_scores.get(did)
        fused.append({
            "content": ref['content'], "score": combined,
            "doc_id": did, "problem": ref['problem'], "action": ref['action']
        })

    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:top_k]


def retrieve_hmr(
    query: str, embedder,
    chunk_index: VectorIndex,
    bm25_scorer: BM25Scorer,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    层次化多粒度检索 v2 (HMR):
    Level 1: BM25 doc-level (lexical, coarse topic filtering)
    Level 2: Dense chunk-level (semantic, fine-grained matching)
    自适应融合: 根据 BM25 分数分布调节 beta
    """
    q_emb = embedder.embed_text(query)

    # Level 1: BM25 doc-level 评分 (所有文档)
    doc_bm25 = bm25_scorer.score_query(query)

    # Min-max 归一化 BM25 分数
    bm25_vals = list(doc_bm25.values())
    bm25_max = max(bm25_vals) if bm25_vals else 1.0
    bm25_min = min(bm25_vals) if bm25_vals else 0.0
    bm25_range = bm25_max - bm25_min
    if bm25_range > 0:
        doc_bm25_normed = {k: (v - bm25_min) / bm25_range for k, v in doc_bm25.items()}
    else:
        doc_bm25_normed = {k: 0.0 for k in doc_bm25}

    # Level 2: Dense chunk 检索 (固定检索池大小, 保证 top-k 一致性)
    _POOL_SIZE = TOP_K * 5  # 固定 = 25, 不随 top_k 参数变化
    chunk_results = chunk_index.search(q_emb, k=max(_POOL_SIZE, top_k))

    # 自适应 beta: BM25 分数标准差高 → 词汇信号可靠 → beta 大
    normed_vals = list(doc_bm25_normed.values())
    bm25_std = float(np.std(normed_vals)) if len(normed_vals) > 1 else 0.0
    beta = min(0.35, max(0.10, bm25_std * 2.5))
    alpha = 1.0 - beta

    # 层次融合
    fused = []
    for item, cs in chunk_results:
        did = item.metadata.get('doc_id', '')
        bm25_s = doc_bm25_normed.get(did, 0.0)
        fused.append({
            "content": item.content,
            "score": alpha * cs + beta * bm25_s,
            "doc_id": did,
            "problem": item.metadata.get("problem", ""),
            "action": item.metadata.get("action", "")
        })

    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:top_k]


def _apply_entity_rerank(
    candidates: List[Dict],
    entity_kb: EntityKnowledgeBase,
    query: str,
    gate_threshold_std: float = 0.15,
    gate_threshold_gap: float = 0.20,
) -> Tuple[List[Dict], bool]:
    """
    实体置信度门控重排序 (三重门控)
    Gate 1: 语义完整性 — 查询必须同时包含组件和症状 (排除仅有类别的泛化查询)
    Gate 2: 实体分数标准差足够大 — 排除无区分度信号
    Gate 3: 实体分数最大-均值差距足够大 — 排除均匀分布

    设计思路: 仅当实体具有足够特异性和区分度时才应用重排
    例: "magneto leaking" → comp={magneto} + sym={leaking} → 通过 (高特异性)
        "HVAC issue" → comp={hvac} + sym={} → 阻止 (缺症状, 无法精确匹配)

    Returns:
        (reranked_candidates, entity_applied: bool)
    """
    q_comps, q_acts, q_syms = entity_kb.extract_from_query(query)

    # Gate 1: 语义完整性 — 必须同时匹配组件 AND 症状
    # 仅有组件 (如 "HVAC") 缺乏足够特异性来区分候选文档
    # 组件+症状 (如 "magneto leaking") 才能精确定位相关维修记录
    if not (q_comps and q_syms):
        return candidates, False

    # 计算实体分数
    entity_scores = [
        entity_kb.compute_entity_score(q_comps, q_acts, q_syms, r['doc_id'])
        for r in candidates
    ]
    max_es = max(entity_scores) if entity_scores else 0.0

    if max_es <= 0:
        return candidates, False

    # 归一化
    normed = [es / max_es for es in entity_scores]
    es_std = float(np.std(normed))
    es_mean = float(np.mean(normed))
    es_max_gap = 1.0 - es_mean

    # Gate 2 & 3: 区分度检查
    if es_std < gate_threshold_std or es_max_gap < gate_threshold_gap:
        return candidates, False

    # 自适应 gamma: 区分度越高 → gamma 越大
    gamma = min(0.30, max(0.08, es_std * 2.0))

    for r, ne in zip(candidates, normed):
        r['score'] = (1 - gamma) * r['score'] + gamma * ne

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates, True


def retrieve_hea_mrag(
    query: str, embedder,
    chunk_index: VectorIndex,
    bm25_scorer: BM25Scorer,
    entity_kb: EntityKnowledgeBase,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    HEA-MRAG: 层次化检索 + 置信度门控实体重排
    Step 1: HMR 获取 3x 候选
    Step 2: 置信度门控 IDF 实体重排
    """
    candidates = retrieve_hmr(query, embedder, chunk_index, bm25_scorer, top_k=top_k * 3)
    candidates, _ = _apply_entity_rerank(candidates, entity_kb, query)
    return candidates[:top_k]


def retrieve_rrf(
    query: str, embedder,
    bm25_scorer: BM25Scorer,
    records: List[MaintenanceRecord],
    chunk_index: VectorIndex,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009)
    RRF(d) = Σ_r 1/(k + rank_r(d))
    将 BM25 和 Dense 的排序列表通过倒数排名融合
    """
    pool = top_k * 10  # 检索更多候选再融合
    bm25_results = retrieve_bm25(query, bm25_scorer, records, top_k=pool)
    dense_results = retrieve_dense(query, embedder, chunk_index, top_k=pool)

    score_map: Dict[str, float] = {}
    ref_map: Dict[str, Dict] = {}

    for rank, r in enumerate(bm25_results):
        did = r['doc_id']
        score_map[did] = score_map.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        if did not in ref_map:
            ref_map[did] = r

    for rank, r in enumerate(dense_results):
        did = r['doc_id']
        score_map[did] = score_map.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        if did not in ref_map:
            ref_map[did] = r

    fused = []
    for did, score in score_map.items():
        ref = ref_map[did]
        fused.append({
            "content": ref['content'], "score": score,
            "doc_id": did, "problem": ref['problem'], "action": ref['action']
        })

    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:top_k]


def retrieve_hyde(
    query: str, embedder,
    chunk_index: VectorIndex,
    hyde_doc: str,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    HyDE: Hypothetical Document Embeddings (Gao et al., 2022)
    使用 LLM 生成假设性文档，然后用假设文档的嵌入进行检索
    """
    # 将查询和假设文档拼接后嵌入 (增强语义)
    hyde_text = f"{query}\n{hyde_doc}"
    hyde_emb = embedder.embed_text(hyde_text)
    results = chunk_index.search(hyde_emb, k=top_k)
    return [
        {"content": item.content, "score": score,
         "doc_id": item.metadata.get("doc_id", ""),
         "problem": item.metadata.get("problem", ""),
         "action": item.metadata.get("action", "")}
        for item, score in results
    ]


# =====================================================================
# 7. 评估指标
# =====================================================================

def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens = set(_normalize(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_hit_at_k(results: List[Dict], relevant_ids: List[str], k: int = TOP_K) -> float:
    retrieved_ids = {r['doc_id'] for r in results[:k]}
    return 1.0 if retrieved_ids & set(relevant_ids) else 0.0


def compute_mrr(results: List[Dict], answer: str, relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for i, r in enumerate(results):
        if r['doc_id'] in relevant_set:
            return 1.0 / (i + 1)
        if _token_f1(r['action'], answer) > 0.3:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(results: List[Dict], answer: str, relevant_ids: List[str], k: int = TOP_K) -> float:
    relevant_set = set(relevant_ids)
    rels = []
    for r in results[:k]:
        if r['doc_id'] in relevant_set:
            rels.append(1.0)
        else:
            rels.append(_token_f1(r['action'], answer))

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(results: List[Dict], answer: str, relevant_ids: List[str], k: int = TOP_K) -> float:
    relevant_set = set(relevant_ids)
    for r in results[:k]:
        if r['doc_id'] in relevant_set:
            return 1.0
        if _token_f1(r['action'], answer) > 0.3:
            return 1.0
    return 0.0


def compute_top1_f1(results: List[Dict], answer: str) -> float:
    if not results:
        return 0.0
    return _token_f1(results[0]['action'], answer)


# =====================================================================
# 8. 实验运行
# =====================================================================

METHODS = ['BM25', 'Dense', 'Hybrid', 'RRF', 'HyDE', 'Ours-HMR', 'Ours-Full']
METRICS = ['mrr', 'ndcg', 'recall', 'hit', 'f1']


def run_domain_experiment(
    domain: str,
    kb_records: List[MaintenanceRecord],
    qa_pairs: List[QAPair],
    chunk_index: VectorIndex,
    bm25_scorer: BM25Scorer,
    entity_kb: EntityKnowledgeBase,
    embedder,
    hyde_docs: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """运行单域实验"""
    results = {m: {metric: [] for metric in METRICS} for m in METHODS}
    entity_applied_count = 0

    for qi, qa in enumerate(qa_pairs):
        if (qi + 1) % 100 == 0:
            print(f"    query {qi+1}/{len(qa_pairs)}", flush=True)

        # --- BM25 ---
        bm25 = retrieve_bm25(qa.query, bm25_scorer, kb_records)
        # --- Dense ---
        dense = retrieve_dense(qa.query, embedder, chunk_index)
        # --- Hybrid ---
        hybrid = retrieve_hybrid(qa.query, embedder, bm25_scorer, kb_records, chunk_index)
        # --- RRF ---
        rrf = retrieve_rrf(qa.query, embedder, bm25_scorer, kb_records, chunk_index)
        # --- HyDE ---
        if hyde_docs is not None:
            hyde = retrieve_hyde(qa.query, embedder, chunk_index, hyde_docs[qi])
        else:
            hyde = dense  # fallback: 无 LLM 时退化为 Dense
        # --- HMR ---
        hmr = retrieve_hmr(qa.query, embedder, chunk_index, bm25_scorer)
        # --- HEA-MRAG ---
        full_candidates = retrieve_hmr(qa.query, embedder, chunk_index, bm25_scorer, top_k=TOP_K * 3)
        full_candidates, applied = _apply_entity_rerank(full_candidates, entity_kb, qa.query)
        full = full_candidates[:TOP_K]
        if applied:
            entity_applied_count += 1

        for method_name, ret in [('BM25', bm25), ('Dense', dense),
                                   ('Hybrid', hybrid), ('RRF', rrf),
                                   ('HyDE', hyde), ('Ours-HMR', hmr),
                                   ('Ours-Full', full)]:
            results[method_name]['mrr'].append(
                compute_mrr(ret, qa.answer, qa.relevant_kb_ids))
            results[method_name]['ndcg'].append(
                compute_ndcg(ret, qa.answer, qa.relevant_kb_ids))
            results[method_name]['recall'].append(
                compute_recall_at_k(ret, qa.answer, qa.relevant_kb_ids))
            results[method_name]['hit'].append(
                compute_hit_at_k(ret, qa.relevant_kb_ids))
            results[method_name]['f1'].append(
                compute_top1_f1(ret, qa.answer))

    print(f"    Entity gate applied: {entity_applied_count}/{len(qa_pairs)} "
          f"({100*entity_applied_count/max(len(qa_pairs),1):.1f}%)")
    return results


def run_ablation(
    qa_pairs: List[QAPair],
    kb_records: List[MaintenanceRecord],
    chunk_index: VectorIndex,
    bm25_scorer: BM25Scorer,
    entity_kb: EntityKnowledgeBase,
    embedder
) -> Dict[str, Dict[str, float]]:
    """消融实验"""
    def _dense_with_entity(q):
        """Dense + Entity (无 HMR)"""
        candidates = retrieve_dense(q, embedder, chunk_index, top_k=TOP_K * 3)
        candidates, _ = _apply_entity_rerank(candidates, entity_kb, q)
        return candidates[:TOP_K]

    configs = {
        'Full (HMR+Entity)': lambda q: retrieve_hea_mrag(q, embedder, chunk_index, bm25_scorer, entity_kb),
        'w/o Entity (HMR only)': lambda q: retrieve_hmr(q, embedder, chunk_index, bm25_scorer),
        'w/o HMR (Dense+Entity)': _dense_with_entity,
        'w/o Both (Dense)': lambda q: retrieve_dense(q, embedder, chunk_index),
    }

    results = {}
    for name, retriever in configs.items():
        mrr_list, ndcg_list, recall_list = [], [], []
        for qa in qa_pairs:
            ret = retriever(qa.query)
            mrr_list.append(compute_mrr(ret, qa.answer, qa.relevant_kb_ids))
            ndcg_list.append(compute_ndcg(ret, qa.answer, qa.relevant_kb_ids))
            recall_list.append(compute_recall_at_k(ret, qa.answer, qa.relevant_kb_ids))

        results[name] = {
            'mrr': float(np.mean(mrr_list)),
            'ndcg': float(np.mean(ndcg_list)),
            'recall': float(np.mean(recall_list)),
        }
    return results


# =====================================================================
# 9. 主函数
# =====================================================================

def main():
    print("=" * 65)
    print("  MaintNet 实验 v4: HEA-MRAG vs Baselines (含 RRF + HyDE)")
    print("  改进: 全量测试 + HMR v2 + 实体门控 + RRF/HyDE 强基线")
    print("=" * 65)

    # 1. 加载并切分数据
    split_data = load_and_split_data()

    # 2. 生成 QA 对
    domain_data = generate_qa_pairs(split_data)

    # 3. 初始化嵌入器
    print("\n初始化嵌入模型 (BGE-small-zh-v1.5)...")
    embedder = get_local_embedder()

    # 4. 构建索引 (仅 KB 部分)
    domain_kb = {d: kb for d, (kb, _) in domain_data.items()}
    chunk_indices = build_chunk_indices(domain_kb, embedder)

    # 5. 构建 BM25 评分器
    bm25_scorers = build_bm25_scorers(domain_kb)

    # 6. 构建实体知识库
    entity_stores = build_entity_stores(domain_kb)

    # 7. HyDE: 用 Qwen2.5-1.5B 生成假设文档
    print("\n[4.5/8] 生成 HyDE 假设文档 (Qwen2.5-1.5B)...")
    try:
        llm_model, llm_tokenizer = load_qwen_llm(device="auto")
        qa_pairs_by_domain = {d: qa for d, (_, qa) in domain_data.items()}
        hyde_docs_by_domain = batch_generate_hyde_docs(qa_pairs_by_domain, llm_model, llm_tokenizer)
        # 释放 LLM 显存
        del llm_model, llm_tokenizer
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("  HyDE 生成完成，LLM 已释放")
    except Exception as e:
        print(f"  警告: HyDE 生成失败 ({e}), 使用 Dense 回退")
        hyde_docs_by_domain = None

    # 8. 运行实验
    print("\n[5/8] 运行检索实验...")
    all_results = {}
    all_ablation = {}

    for domain in domain_data:
        kb_records, qa_pairs = domain_data[domain]
        chunk_idx = chunk_indices[domain]
        bm25 = bm25_scorers[domain]
        ekb = entity_stores[domain]
        hyde_docs = hyde_docs_by_domain.get(domain) if hyde_docs_by_domain else None

        print(f"\n  === {domain.upper()} (n={len(qa_pairs)}) ===")
        results = run_domain_experiment(
            domain, kb_records, qa_pairs,
            chunk_idx, bm25, ekb, embedder,
            hyde_docs=hyde_docs
        )
        all_results[domain] = results

        print(f"  消融实验...")
        ablation = run_ablation(
            qa_pairs, kb_records, chunk_idx, bm25, ekb, embedder
        )
        all_ablation[domain] = ablation

    # 9. 统计显著性
    print("\n[6/8] 统计显著性检验...")
    significance = {}
    for domain in domain_data:
        r = all_results[domain]
        sig_tests = []
        for base in ['Dense', 'BM25', 'Hybrid', 'RRF', 'HyDE']:
            for ours in ['Ours-Full', 'Ours-HMR']:
                if len(r[ours]['mrr']) == len(r[base]['mrr']):
                    t, p = stats.ttest_rel(r[ours]['mrr'], r[base]['mrr'])
                    p_val = float(p) if not np.isnan(p) else 1.0
                    sig_tests.append({
                        'comparison': f"{ours} vs {base}",
                        'p_value': p_val,
                        't_stat': float(t) if not np.isnan(t) else 0.0,
                        'significant': bool(p_val < 0.05),
                        'mean_diff': float(np.mean(r[ours]['mrr']) - np.mean(r[base]['mrr']))
                    })
        significance[domain] = sig_tests

    # ============= 输出 =============
    print("\n" + "=" * 65)
    print("  实验结果汇总")
    print("=" * 65)

    for domain in domain_data:
        n = len(domain_data[domain][1])
        print(f"\n【{domain.upper()}】 (n={n})")
        print(f"{'Method':<14} {'MRR':>7} {'NDCG@5':>7} {'Recall':>7} {'Hit@5':>7} {'F1':>7}")
        print("-" * 55)

        r = all_results[domain]
        for m in METHODS:
            vals = {metric: np.mean(r[m][metric]) for metric in METRICS}
            print(f"{m:<14} {vals['mrr']:>7.3f} {vals['ndcg']:>7.3f} "
                  f"{vals['recall']:>7.3f} {vals['hit']:>7.3f} {vals['f1']:>7.3f}")

    # 消融
    print("\n" + "=" * 65)
    print("  消融实验")
    print("=" * 65)
    for domain in domain_data:
        print(f"\n【{domain.upper()}】")
        print(f"  {'Config':<25} {'MRR':>7} {'NDCG':>7} {'Recall':>7}")
        print(f"  {'-'*50}")
        for name, m in all_ablation[domain].items():
            print(f"  {name:<25} {m['mrr']:>7.3f} {m['ndcg']:>7.3f} {m['recall']:>7.3f}")

    # 显著性
    print("\n" + "=" * 65)
    print("  统计显著性 (Paired t-test)")
    print("=" * 65)
    for domain in domain_data:
        print(f"\n【{domain.upper()}】")
        for s in significance[domain]:
            star = "***" if s['p_value'] < 0.01 else ("*" if s['p_value'] < 0.05 else "n.s.")
            print(f"  {s['comparison']:<28} p={s['p_value']:.4f} {star:>4}  Δ={s['mean_diff']:+.3f}")

    # 跨域平均
    print("\n" + "=" * 65)
    print("  跨域平均")
    print("=" * 65)
    domains = list(domain_data.keys())
    print(f"{'Method':<14} {'MRR':>7} {'NDCG@5':>7} {'Recall':>7} {'Hit@5':>7} {'F1':>7}")
    print("-" * 55)
    for m in METHODS:
        avgs = {}
        for metric in METRICS:
            avgs[metric] = np.mean([np.mean(all_results[d][m][metric]) for d in domains])
        print(f"{m:<14} {avgs['mrr']:>7.3f} {avgs['ndcg']:>7.3f} "
              f"{avgs['recall']:>7.3f} {avgs['hit']:>7.3f} {avgs['f1']:>7.3f}")

    # 保存
    print("\n[7/8] 保存结果...")
    output = {
        'main_results': {
            d: {m: {metric: float(np.mean(all_results[d][m][metric]))
                    for metric in METRICS}
                for m in METHODS}
            for d in domains
        },
        'ablation': {d: {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in abl.items()}
                     for d, abl in all_ablation.items()},
        'significance': significance,
        'config': {'top_k': TOP_K, 'seed': RANDOM_SEED,
                   'train_ratio': TRAIN_RATIO,
                   'num_queries': 'all' if NUM_QUERIES == 0 else NUM_QUERIES,
                   'rrf_k': RRF_K,
                   'hyde_model': 'Qwen2.5-1.5B-Instruct'}
    }
    out_path = OUTPUT_DIR / "maintnet_results_v4.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {out_path}")
    print("\n实验完成!")


if __name__ == "__main__":
    main()
