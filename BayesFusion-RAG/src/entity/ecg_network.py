"""
学习式实体置信门控网络 (ECGNet)
将规则式门控替换为轻量神经网络,自动学习何时启用实体增强
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class ECGFeatures:
    """ECG 网络输入特征"""
    query_embedding: np.ndarray  # 512维
    has_component: float         # 0/1
    has_symptom: float           # 0/1
    has_action: float            # 0/1
    num_matched_entities: float  # 归一化
    avg_entity_idf: float
    entity_score_std: float
    top1_score: float
    score_std: float
    score_gap: float             # top1 - mean
    score_entropy: float
    top5_mean: float
    query_type: int              # 0-3


class ECGNet(nn.Module):
    """
    轻量门控网络 (~105K参数)
    输入: 527维 (512 query_emb + 6 entity_feat + 5 score_feat + 4 query_type_onehot)
    输出: 2维 [gate_prob, gamma]
    """
    def __init__(self, input_dim: int = 527, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.gate_branch = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.gamma_branch = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"ECGNet: {total_params:,} params (~{total_params/1000:.1f}K)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_proj(x)
        gate_prob = self.gate_branch(features).squeeze(-1)
        gamma = self.gamma_branch(features).squeeze(-1) * 0.4
        return gate_prob, gamma


class ECGFeatureExtractor:
    """特征提取器"""

    QUERY_TYPE_MAP = {'component_symptom': 0, 'procedure': 1, 'specification': 2, 'general': 3}

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

    def __init__(self, embedder, entity_kb):
        self.embedder = embedder
        self.entity_kb = entity_kb

    def extract_features(self, query: str, candidates: List[Dict]) -> ECGFeatures:
        query_embedding = np.array(self.embedder.embed_text(query), dtype=np.float32)

        q_comps, q_acts, q_syms = self._extract_from_query(query)
        has_component = 1.0 if q_comps else 0.0
        has_symptom = 1.0 if q_syms else 0.0
        has_action = 1.0 if q_acts else 0.0

        num_entities = len(q_comps | q_acts | q_syms)
        avg_idf = 0.0
        if num_entities > 0:
            avg_idf = sum(self.entity_kb.get_idf(e) for e in (q_comps | q_acts | q_syms)) / num_entities

        entity_scores = []
        for r in candidates:
            doc_comps, doc_acts, doc_syms = self.entity_kb.get_doc_entities(r.get('doc_id', ''))
            score = self._compute_entity_score(q_comps, q_acts, q_syms, doc_comps, doc_acts, doc_syms)
            entity_scores.append(score)

        max_es = max(entity_scores) if entity_scores else 1.0
        normed_es = [es / max_es if max_es > 0 else 0.0 for es in entity_scores]
        entity_score_std = float(np.std(normed_es)) if normed_es else 0.0

        scores = [r.get('score', 0) for r in candidates]
        if scores:
            score_std = float(np.std(scores))
            score_gap = scores[0] - float(np.mean(scores))
            probs = np.array(scores) / (sum(scores) + 1e-10)
            score_entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
            top1_score = scores[0]
            top5_mean = float(np.mean(scores[:5]))
        else:
            score_std, score_gap, score_entropy, top1_score, top5_mean = 0, 0, 0, 0, 0

        query_type = self.QUERY_TYPE_MAP[self.classify_query_type(query)]

        return ECGFeatures(
            query_embedding=query_embedding,
            has_component=has_component, has_symptom=has_symptom, has_action=has_action,
            num_matched_entities=min(num_entities / 10.0, 1.0),
            avg_entity_idf=avg_idf, entity_score_std=entity_score_std,
            top1_score=top1_score, score_std=score_std, score_gap=score_gap,
            score_entropy=score_entropy, top5_mean=top5_mean, query_type=query_type
        )

    def classify_query_type(self, query: str) -> str:
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
        q = query.lower()
        comps = {c for c in self.COMPONENTS if c in q}
        acts = {a for a in self.ACTIONS if a in q}
        syms = {s for s in self.SYMPTOMS if s in q}
        return comps, acts, syms

    def _compute_entity_score(self, q_comps, q_acts, q_syms, doc_comps, doc_acts, doc_syms) -> float:
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

    def to_tensor(self, features: ECGFeatures) -> torch.Tensor:
        query_type_onehot = np.zeros(4, dtype=np.float32)
        query_type_onehot[features.query_type] = 1.0

        feature_vector = np.concatenate([
            features.query_embedding,
            np.array([features.has_component, features.has_symptom, features.has_action,
                      features.num_matched_entities, features.avg_entity_idf, features.entity_score_std], dtype=np.float32),
            np.array([features.top1_score, features.score_std, features.score_gap,
                      features.score_entropy, features.top5_mean], dtype=np.float32),
            query_type_onehot
        ])
        return torch.from_numpy(feature_vector)


class ECGLoss(nn.Module):
    def __init__(self, gate_weight: float = 0.6, gamma_weight: float = 0.4):
        super().__init__()
        self.gate_weight = gate_weight
        self.gamma_weight = gamma_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_pred, gamma_pred = pred[:, 0], pred[:, 1]
        gate_label, gamma_label = target[:, 0], target[:, 1]

        gate_loss = F.binary_cross_entropy(gate_pred, gate_label)
        gamma_mask = (gate_label > 0.5).float()
        gamma_loss = F.mse_loss(gamma_pred * gamma_mask, gamma_label * gamma_mask) if gamma_mask.sum() > 0 else torch.tensor(0.0)

        return {'total': self.gate_weight * gate_loss + self.gamma_weight * gamma_loss, 'gate': gate_loss, 'gamma': gamma_loss}


class ECGTrainer:
    def __init__(self, model: ECGNet, device: str = "cuda:0"):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = ECGLoss()

    def train(self, train_data: List[Dict], val_data: List[Dict], epochs: int = 50, lr: float = 1e-3, batch_size: int = 32) -> Dict:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': [], 'gate_acc': []}

        for epoch in range(epochs):
            self.model.train()
            train_losses = []

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                features = torch.stack([d['features'] for d in batch]).to(self.device)
                labels = torch.tensor([[d['gate_label'], d['gamma_label']] for d in batch], dtype=torch.float32).to(self.device)

                gate_pred, gamma_pred = self.model(features)
                pred = torch.stack([gate_pred, gamma_pred], dim=1)
                losses = self.loss_fn(pred, labels)

                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
                train_losses.append(losses['total'].item())

            val_metrics = self.evaluate(val_data, batch_size)
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_metrics['loss'])
            history['gate_acc'].append(val_metrics['gate_accuracy'])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Val Loss={val_metrics['loss']:.4f}, Gate Acc={val_metrics['gate_accuracy']:.3f}")

        return history

    def evaluate(self, test_data: List[Dict], batch_size: int = 32) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                features = torch.stack([d['features'] for d in batch]).to(self.device)
                labels = torch.tensor([[d['gate_label'], d['gamma_label']] for d in batch], dtype=torch.float32)

                gate_pred, _ = self.model(features)
                all_preds.extend(gate_pred.cpu().numpy())
                all_labels.extend(labels[:, 0].numpy())

        gate_acc = float(np.mean((np.array(all_preds) > 0.5) == (np.array(all_labels) > 0.5)))
        return {'loss': 0.0, 'gate_accuracy': gate_acc, 'gamma_mae': 0.0}

    def save_checkpoint(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def apply_learned_entity_gate(candidates: List[Dict], ecg_net: ECGNet, feature_extractor: ECGFeatureExtractor,
                               query: str, device: str = "cuda:0") -> Tuple[List[Dict], bool, float]:
    ecg_net.eval()
    features = feature_extractor.extract_features(query, candidates)
    feature_tensor = feature_extractor.to_tensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        gate_prob, gamma = ecg_net(feature_tensor)
        gate_prob, gamma = gate_prob.item(), gamma.item()

    if gate_prob < 0.5:
        return candidates, False, 0.0

    q_comps, q_acts, q_syms = feature_extractor._extract_from_query(query)
    entity_scores = []
    for r in candidates:
        doc_comps, doc_acts, doc_syms = feature_extractor.entity_kb.get_doc_entities(r.get('doc_id', ''))
        score = feature_extractor._compute_entity_score(q_comps, q_acts, q_syms, doc_comps, doc_acts, doc_syms)
        entity_scores.append(score)

    max_es = max(entity_scores) if entity_scores else 1.0
    for r, es in zip(candidates, entity_scores):
        r['score'] = (1 - gamma) * r['score'] + gamma * (es / max_es if max_es > 0 else 0)

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates, True, gamma
