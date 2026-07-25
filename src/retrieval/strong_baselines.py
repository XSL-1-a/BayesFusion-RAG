"""
强基线检索器模块
包含 BGE-Reranker 和 ColBERT 近似实现
"""
import torch
import numpy as np
from typing import List, Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    BGE-Reranker: Cross-Encoder 重排序器
    使用 ModelScope 镜像: AI-ModelScope/bge-reranker-base
    """
    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        """
        初始化 BGE Reranker

        Args:
            model_path: 本地模型路径
            device: 设备 (cuda:0, cuda:1, cpu)
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.model = None
        self.tokenizer = None

        # 确定模型路径
        if model_path is None:
            model_path = os.environ.get('BGE_RERANKER_PATH')
            if model_path is None:
                default_path = "./models/AI-ModelScope/bge-reranker-base"
                if os.path.exists(default_path):
                    model_path = default_path
                else:
                    model_path = "BAAI/bge-reranker-base"

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """加载模型和分词器"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading BGE-Reranker from {self.model_path} on {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info("BGE-Reranker loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BGE-Reranker: {e}")
            raise

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """计算查询和文档的相关性分数"""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

            if isinstance(scores, np.ndarray) and scores.ndim == 0:
                scores = [float(scores)]
            else:
                scores = scores.tolist()

        return scores

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """对检索结果进行重排序"""
        if not documents:
            return []

        doc_texts = [doc.get('content', '') or doc.get('problem', '') + ' ' + doc.get('action', '')
                     for doc in documents]

        rerank_scores = self.compute_scores(query, doc_texts)

        reranked = []
        for doc, score in zip(documents, rerank_scores):
            new_doc = doc.copy()
            new_doc['rerank_score'] = float(score)
            new_doc['original_score'] = doc.get('score', 0.0)
            new_doc['score'] = float(score)
            reranked.append(new_doc)

        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked[:top_k]


class ColBERTRetriever:
    """
    ColBERT 近似实现: 基于 Token 级 MaxSim
    """
    def __init__(self, embedder, max_query_tokens: int = 32, max_doc_tokens: int = 180):
        """
        初始化 ColBERT 检索器

        Args:
            embedder: 嵌入器实例 (HuggingFaceEmbedder)
            max_query_tokens: 查询最大 token 数
            max_doc_tokens: 文档最大 token 数
        """
        self.embedder = embedder
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self._doc_token_cache: Dict[str, np.ndarray] = {}

    def _encode_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        """将文本编码为 token 级嵌入"""
        tokens = text.lower().split()[:max_tokens]
        embeddings = []
        for token in tokens:
            emb = self.embedder.embed_text(token)
            embeddings.append(emb)

        if not embeddings:
            return np.zeros((1, self.embedder.dimension), dtype=np.float32)

        result = np.array(embeddings, dtype=np.float32)
        # 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-10)
        return result

    def _maxsim(self, query_embs: np.ndarray, doc_embs: np.ndarray) -> float:
        """
        计算 MaxSim 分数
        MaxSim(Q, D) = Σ_{q_i ∈ Q} max_{d_j ∈ D} sim(q_i, d_j)
        """
        sim_matrix = np.dot(query_embs, doc_embs.T)
        max_sims = np.max(sim_matrix, axis=1)
        return float(np.sum(max_sims))

    def retrieve(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """ColBERT 检索/重排序"""
        if not candidates:
            return []

        query_embs = self._encode_tokens(query, self.max_query_tokens)

        scored = []
        for doc in candidates:
            doc_id = doc.get('doc_id', '')

            if doc_id and doc_id in self._doc_token_cache:
                doc_embs = self._doc_token_cache[doc_id]
            else:
                doc_text = doc.get('content', '') or doc.get('problem', '') + ' ' + doc.get('action', '')
                doc_embs = self._encode_tokens(doc_text, self.max_doc_tokens)
                if doc_id:
                    self._doc_token_cache[doc_id] = doc_embs

            maxsim_score = self._maxsim(query_embs, doc_embs)

            new_doc = doc.copy()
            new_doc['colbert_score'] = maxsim_score
            new_doc['original_score'] = doc.get('score', 0.0)
            new_doc['score'] = maxsim_score
            scored.append(new_doc)

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def clear_cache(self):
        """清除文档 token 缓存"""
        self._doc_token_cache.clear()


class CrossEncoderReranker:
    """
    通用 Cross-Encoder 重排序器
    支持 sentence-transformers 的 cross-encoder 模型
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("CrossEncoder loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CrossEncoder: {e}")
            raise

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """计算相关性分数"""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """重排序文档"""
        if not documents:
            return []

        doc_texts = [doc.get('content', '') or doc.get('problem', '') + ' ' + doc.get('action', '')
                     for doc in documents]

        scores = self.compute_scores(query, doc_texts)

        reranked = []
        for doc, score in zip(documents, scores):
            new_doc = doc.copy()
            new_doc['rerank_score'] = float(score)
            new_doc['original_score'] = doc.get('score', 0.0)
            new_doc['score'] = float(score)
            reranked.append(new_doc)

        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked[:top_k]
