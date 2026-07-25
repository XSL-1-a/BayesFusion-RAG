"""
MoRE (Mixture-of-Retrieval-Experts) 门控网络
学习 query → expert weights 的映射，实现查询自适应的专家路由

核心设计:
- 输入: 查询特征向量 (feature_dim)
- 输出: N维专家权重向量 (在单纯形上, 通过softmax)
- 损失函数:
  1. 任务损失: 融合结果的排序质量 (ListNet / NDCG proxy)
  2. 负载均衡损失: Switch Transformer 风格，防止专家塌缩
  3. 专家化损失: 鼓励不同查询激活不同专家

理论基础:
- N-Expert 最优权重理论: w*_i = (ρ_i / σ²_i) / Σ_j (ρ_j / σ²_j)
- 门控网络隐式学习逼近该理论最优

作者: RAG Research Team
日期: 2026-01-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GatingOutput:
    """门控网络输出"""
    expert_weights: torch.Tensor    # (batch, n_experts) 在单纯形上
    expert_logits: torch.Tensor     # (batch, n_experts) softmax前的logits
    top_expert_idx: torch.Tensor    # (batch,) 最大权重专家索引


class MoREGatingNet(nn.Module):
    """
    MoRE 门控网络

    架构: feature_dim → hidden → hidden/2 → n_experts (softmax)

    支持两种输出模式:
    - soft: softmax 归一化权重 (默认)
    - top_k: 仅保留 top-k 专家，其余置零后重归一化

    参数量约: feature_dim * hidden + hidden * hidden/2 + hidden/2 * n_experts
    """

    def __init__(
        self,
        feature_dim: int = 3090,
        n_experts: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
        top_k: int = 0
    ):
        """
        Args:
            feature_dim: 输入特征维度
            n_experts: 专家数量
            hidden_dim: 隐藏层维度
            dropout: Dropout 比率
            temperature: softmax 温度 (越低越 sparse)
            top_k: 若 > 0 则仅保留 top-k 专家
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.top_k = top_k

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_experts)
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"MoREGatingNet: {total_params:,} params (~{total_params/1000:.1f}K), "
              f"n_experts={n_experts}, feature_dim={feature_dim}")

    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> GatingOutput:
        """
        前向传播

        Args:
            x: (batch, feature_dim) 查询特征

        Returns:
            GatingOutput 包含权重、logits和top专家索引
        """
        h = self.feature_proj(x)
        logits = self.gate_head(h)  # (batch, n_experts)

        # 温度缩放 softmax
        weights = F.softmax(logits / self.temperature, dim=-1)

        # Top-k 稀疏化
        if self.top_k > 0 and self.top_k < self.n_experts:
            topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_idx, 1.0)
            weights = weights * mask
            # 重归一化
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

        top_expert_idx = torch.argmax(weights, dim=-1)

        return GatingOutput(
            expert_weights=weights,
            expert_logits=logits,
            top_expert_idx=top_expert_idx
        )


class MoRELoss(nn.Module):
    """
    MoRE 复合损失函数

    L_total = L_task + λ_balance × L_balance + λ_spec × L_specialization

    1. L_task: 基于 NDCG proxy 的排序损失 (ListNet approx)
       - 使门控网络学习产生使融合排序最优的权重

    2. L_balance: Switch Transformer 风格负载均衡
       - L_balance = N × Σ_i (f_i × P_i)
       - f_i = 专家i被选为top的频率, P_i = 平均路由概率
       - 防止所有查询路由到同一个专家

    3. L_specialization: 专家化损失
       - 鼓励单个查询的权重分布具有低熵 (确定性路由)
       - 同时鼓励不同查询激活不同专家 (跨查询多样性)
    """

    def __init__(
        self,
        n_experts: int = 4,
        lambda_balance: float = 0.1,
        lambda_spec: float = 0.01,
        lambda_entropy: float = 0.05
    ):
        """
        Args:
            n_experts: 专家数量
            lambda_balance: 负载均衡损失权重
            lambda_spec: 专家化损失权重
            lambda_entropy: 单查询熵损失权重
        """
        super().__init__()
        self.n_experts = n_experts
        self.lambda_balance = lambda_balance
        self.lambda_spec = lambda_spec
        self.lambda_entropy = lambda_entropy

    def forward(
        self,
        expert_weights: torch.Tensor,
        fused_scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        expert_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算复合损失

        Args:
            expert_weights: (batch, n_experts) 门控权重
            fused_scores: (batch, n_docs) 融合后的文档分数
            relevance_labels: (batch, n_docs) 相关性标签
            expert_logits: (batch, n_experts) softmax前的logits (可选)

        Returns:
            包含各项损失的字典
        """
        # 1. 任务损失: ListNet-style KL 散度
        task_loss = self._listnet_loss(fused_scores, relevance_labels)

        # 2. 负载均衡损失
        balance_loss = self._balance_loss(expert_weights)

        # 3. 专家化损失 (单查询低熵)
        entropy_loss = self._entropy_loss(expert_weights)

        # 4. 跨查询专家化 (鼓励不同查询使用不同专家)
        spec_loss = self._specialization_loss(expert_weights)

        # 总损失
        total_loss = (
            task_loss
            + self.lambda_balance * balance_loss
            + self.lambda_entropy * entropy_loss
            + self.lambda_spec * spec_loss
        )

        return {
            'total': total_loss,
            'task': task_loss,
            'balance': balance_loss,
            'entropy': entropy_loss,
            'specialization': spec_loss
        }

    def _listnet_loss(
        self,
        pred_scores: torch.Tensor,
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        ListNet 损失: 预测分数分布与真实相关性分布的 KL 散度

        P_pred = softmax(pred_scores)
        P_target = softmax(target_labels)
        L = -Σ P_target × log(P_pred)
        """
        pred_dist = F.softmax(pred_scores, dim=-1)
        target_dist = F.softmax(target_labels.float(), dim=-1)

        loss = -torch.sum(target_dist * torch.log(pred_dist + 1e-10), dim=-1)
        return loss.mean()

    def _balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer 风格负载均衡损失

        L_balance = N × Σ_i (f_i × P_i)
        - f_i: 专家i被选为 argmax 的频率
        - P_i: 专家i的平均路由概率
        """
        batch_size = expert_weights.size(0)
        n_experts = expert_weights.size(1)

        # f_i: 每个专家被选为top-1的频率
        top_expert = torch.argmax(expert_weights, dim=-1)
        f = torch.zeros(n_experts, device=expert_weights.device)
        for i in range(n_experts):
            f[i] = (top_expert == i).float().sum() / batch_size

        # P_i: 平均路由概率
        P = expert_weights.mean(dim=0)

        # L_balance = N * dot(f, P)
        return n_experts * torch.dot(f, P)

    def _entropy_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        单查询熵损失
        鼓励每个查询的专家权重分布具有低熵 (确定性选择)
        H(w) = -Σ w_i log(w_i)
        """
        entropy = -torch.sum(
            expert_weights * torch.log(expert_weights + 1e-10), dim=-1
        )
        return entropy.mean()

    def _specialization_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        专家化损失
        鼓励不同查询激活不同专家：最大化跨查询的专家使用方差

        对每个专家，计算其在batch中被分配权重的方差
        方差越大 → 有些查询强烈使用该专家，有些不使用 → 更专业化
        取负方差作为损失 (最小化损失 = 最大化方差)
        """
        if expert_weights.size(0) < 2:
            return torch.tensor(0.0, device=expert_weights.device)

        # 每个专家在batch中的权重方差
        expert_var = expert_weights.var(dim=0)  # (n_experts,)

        # 取负均值方差作为损失
        return -expert_var.mean()


class MoRETrainer:
    """
    MoRE 门控网络训练器

    训练流程:
    1. 对每个训练样本，提取查询特征
    2. 门控网络预测专家权重
    3. 根据权重加权融合各专家的检索分数
    4. 计算复合损失并反向传播
    """

    def __init__(
        self,
        model: MoREGatingNet,
        n_experts: int = 4,
        device: str = "cpu",
        lambda_balance: float = 0.1,
        lambda_spec: float = 0.01,
        lambda_entropy: float = 0.05
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = MoRELoss(
            n_experts=n_experts,
            lambda_balance=lambda_balance,
            lambda_spec=lambda_spec,
            lambda_entropy=lambda_entropy
        )

    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        patience: int = 15
    ) -> Dict:
        """
        训练门控网络

        Args:
            train_data: 训练数据，每项包含:
                - 'features': Tensor (feature_dim,)
                - 'expert_scores': Tensor (n_experts, n_docs) 各专家分数
                - 'relevance': Tensor (n_docs,) 相关性标签
            val_data: 验证数据 (同上格式)
            epochs: 训练轮次
            lr: 学习率
            batch_size: 批大小
            weight_decay: 权重衰减
            patience: 早停耐心值

        Returns:
            训练历史字典
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        history = {
            'train_loss': [], 'val_loss': [],
            'train_task_loss': [], 'val_task_loss': [],
            'balance_loss': [], 'entropy_loss': [],
            'expert_usage': [], 'val_ndcg': []
        }

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_losses = []
            task_losses = []
            bal_losses = []
            ent_losses = []
            all_weights = []

            indices = np.random.permutation(len(train_data))

            for i in range(0, len(train_data), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = [train_data[j] for j in batch_idx]

                features = torch.stack([d['features'] for d in batch]).to(self.device)
                expert_scores = torch.stack([d['expert_scores'] for d in batch]).to(self.device)
                relevance = torch.stack([d['relevance'] for d in batch]).to(self.device)

                # 前向
                gating_out = self.model(features)
                weights = gating_out.expert_weights  # (batch, n_experts)

                # 加权融合: fused = Σ w_i × expert_scores_i
                # expert_scores: (batch, n_experts, n_docs)
                # weights: (batch, n_experts) → (batch, n_experts, 1)
                fused = torch.sum(
                    weights.unsqueeze(-1) * expert_scores, dim=1
                )  # (batch, n_docs)

                # 计算损失
                losses = self.loss_fn(
                    expert_weights=weights,
                    fused_scores=fused,
                    relevance_labels=relevance,
                    expert_logits=gating_out.expert_logits
                )

                # 反向
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(losses['total'].item())
                task_losses.append(losses['task'].item())
                bal_losses.append(losses['balance'].item())
                ent_losses.append(losses['entropy'].item())
                all_weights.append(weights.detach().cpu().numpy())

            scheduler.step()

            # 记录专家使用统计
            all_weights_np = np.concatenate(all_weights, axis=0)
            expert_usage = all_weights_np.mean(axis=0).tolist()

            # 验证
            val_metrics = self.evaluate(val_data, batch_size)

            # 记录历史
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_metrics['loss'])
            history['train_task_loss'].append(np.mean(task_losses))
            history['val_task_loss'].append(val_metrics['task_loss'])
            history['balance_loss'].append(np.mean(bal_losses))
            history['entropy_loss'].append(np.mean(ent_losses))
            history['expert_usage'].append(expert_usage)
            history['val_ndcg'].append(val_metrics['ndcg@5'])

            # 早停
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                usage_str = ', '.join(f'{u:.3f}' for u in expert_usage)
                print(
                    f"Epoch {epoch+1}: "
                    f"Train={np.mean(train_losses):.4f}, "
                    f"Val={val_metrics['loss']:.4f}, "
                    f"NDCG@5={val_metrics['ndcg@5']:.4f}, "
                    f"Usage=[{usage_str}]"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def evaluate(self, data: List[Dict], batch_size: int = 32) -> Dict:
        """评估模型"""
        self.model.eval()
        all_losses = []
        all_task_losses = []
        all_ndcg = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                features = torch.stack([d['features'] for d in batch]).to(self.device)
                expert_scores = torch.stack([d['expert_scores'] for d in batch]).to(self.device)
                relevance = torch.stack([d['relevance'] for d in batch]).to(self.device)

                gating_out = self.model(features)
                weights = gating_out.expert_weights

                fused = torch.sum(
                    weights.unsqueeze(-1) * expert_scores, dim=1
                )

                losses = self.loss_fn(
                    expert_weights=weights,
                    fused_scores=fused,
                    relevance_labels=relevance
                )

                all_losses.append(losses['total'].item())
                all_task_losses.append(losses['task'].item())

                # 计算 NDCG@5
                for j in range(fused.size(0)):
                    ndcg = self._compute_ndcg(
                        fused[j].cpu().numpy(),
                        relevance[j].cpu().numpy(),
                        k=5
                    )
                    all_ndcg.append(ndcg)

        return {
            'loss': np.mean(all_losses),
            'task_loss': np.mean(all_task_losses),
            'ndcg@5': np.mean(all_ndcg)
        }

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

    def save_checkpoint(self, path: str, history: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'feature_dim': self.model.feature_dim,
                'n_experts': self.model.n_experts,
                'hidden_dim': self.model.hidden_dim,
                'temperature': self.model.temperature,
                'top_k': self.model.top_k
            }
        }
        if history is not None:
            checkpoint['history'] = history
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('history', None)
