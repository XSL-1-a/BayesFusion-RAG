"""
多模态增强检索模块
包含 VLM 摘要、CLIP 对齐和自适应融合

作者: RAG Research Team
日期: 2026-01-28
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
import logging
import json
import os

logger = logging.getLogger(__name__)


class QwenVLSummarizer:
    """
    基于 Qwen2-VL-7B 的技术图像摘要生成器

    功能:
    1. 对维护记录中的技术图像生成结构化描述
    2. 提取图像中的关键信息（组件、状态、损坏类型等）
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda:0",
        use_4bit: bool = False
    ):
        """
        初始化 Qwen2-VL 摘要器

        Args:
            model_path: 模型路径，默认使用 ModelScope 镜像
            device: 设备
            use_4bit: 是否使用 4-bit 量化（节省显存）
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.use_4bit = use_4bit
        self.model = None
        self.processor = None

        if model_path is None:
            model_path = os.environ.get('QWEN_VL_PATH')
            if model_path is None:
                default_path = "./models/Qwen/Qwen2-VL-7B-Instruct"
                if os.path.exists(default_path):
                    model_path = default_path
                else:
                    model_path = "Qwen/Qwen2-VL-7B-Instruct"

        self.model_path = model_path
        self._loaded = False

    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            logger.info(f"Loading Qwen2-VL from {self.model_path}")

            load_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if "cuda" in self.device else None
            }

            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            if not self.use_4bit and "cuda" in self.device:
                self.model = self.model.to(self.device)

            self.model.eval()
            self._loaded = True
            logger.info("Qwen2-VL loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL: {e}")
            raise

    def generate_summary(
        self,
        image_path: str,
        context: str = "",
        max_length: int = 200
    ) -> str:
        """
        为技术图像生成结构化摘要

        Args:
            image_path: 图像路径
            context: 上下文信息（如问题描述）
            max_length: 最大生成长度

        Returns:
            图像摘要文本
        """
        self._load_model()

        if not os.path.exists(image_path):
            return ""

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open image {image_path}: {e}")
            return ""

        prompt = self._build_prompt(context)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device if not self.use_4bit else self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=0.1
            )

        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        summary = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return summary

    def _build_prompt(self, context: str) -> str:
        """构建提示词"""
        base_prompt = """Analyze this technical/maintenance image and provide a structured description:
1. Main component(s) visible
2. Condition/state (normal, damaged, worn, etc.)
3. Any visible defects or issues
4. Relevant technical details

Keep the description concise and factual."""

        if context:
            base_prompt += f"\n\nContext: {context}"

        return base_prompt

    def batch_summarize(
        self,
        image_paths: List[str],
        contexts: List[str] = None,
        batch_size: int = 4
    ) -> List[str]:
        """批量生成摘要"""
        if contexts is None:
            contexts = [""] * len(image_paths)

        summaries = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]

            for path, ctx in zip(batch_paths, batch_contexts):
                summary = self.generate_summary(path, ctx)
                summaries.append(summary)

        return summaries


class CLIPAligner:
    """
    CLIP 图像-文本对齐器

    功能:
    1. 计算图像与查询的语义相似度
    2. 提供跨模态对齐分数
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda:0"
    ):
        """
        初始化 CLIP 对齐器

        Args:
            model_name: CLIP 模型名称
            device: 设备
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False

    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP from {self.model_name}")

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("CLIP loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """编码图像为 CLIP 嵌入"""
        self._load_model()

        if not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open image {image_path}: {e}")
            return None

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu().numpy()[0]

    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为 CLIP 嵌入"""
        self._load_model()

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu().numpy()[0]

    def compute_similarity(
        self,
        image_path: str,
        text: str
    ) -> float:
        """计算图像-文本相似度"""
        image_emb = self.encode_image(image_path)
        if image_emb is None:
            return 0.0

        text_emb = self.encode_text(text)

        similarity = float(np.dot(image_emb, text_emb))
        return similarity

    def batch_encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> Dict[str, np.ndarray]:
        """批量编码图像"""
        self._load_model()

        embeddings = {}

        valid_paths = []
        valid_images = []

        for path in image_paths:
            if os.path.exists(path):
                try:
                    image = Image.open(path).convert("RGB")
                    valid_paths.append(path)
                    valid_images.append(image)
                except Exception:
                    continue

        for i in range(0, len(valid_images), batch_size):
            batch_images = valid_images[i:i+batch_size]
            batch_paths = valid_paths[i:i+batch_size]

            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)

            for path, emb in zip(batch_paths, image_features.cpu().numpy()):
                embeddings[path] = emb

        return embeddings


class AdaptiveMultimodalFusion:
    """
    自适应多模态融合器

    根据查询的图像需求程度动态调整融合权重
    """

    IMAGE_KEYWORDS = {
        'high': ['diagram', 'schematic', 'image', 'photo', 'picture', 'visual',
                 'show', 'look', 'appear', 'see', 'display', 'crack', 'leak',
                 'damage', 'worn', 'broken', 'corrosion', 'rust'],
        'medium': ['part', 'component', 'assembly', 'system', 'location',
                   'position', 'where', 'identify', 'inspect'],
        'low': ['procedure', 'step', 'how', 'specification', 'manual',
                'standard', 'regulation', 'policy']
    }

    WEIGHT_CONFIGS = {
        'high': {'text': 0.40, 'image': 0.35, 'summary': 0.25},
        'medium': {'text': 0.55, 'image': 0.25, 'summary': 0.20},
        'low': {'text': 0.75, 'image': 0.15, 'summary': 0.10}
    }

    def __init__(
        self,
        text_embedder,
        clip_aligner: CLIPAligner,
        vlm_summarizer: Optional[QwenVLSummarizer] = None
    ):
        """
        初始化自适应融合器

        Args:
            text_embedder: 文本嵌入器 (BGE)
            clip_aligner: CLIP 对齐器
            vlm_summarizer: VLM 摘要器（可选）
        """
        self.text_embedder = text_embedder
        self.clip_aligner = clip_aligner
        self.vlm_summarizer = vlm_summarizer

        self._image_cache: Dict[str, np.ndarray] = {}
        self._summary_cache: Dict[str, str] = {}

    def estimate_image_need(self, query: str) -> Tuple[str, float]:
        """
        估计查询的图像需求程度

        Returns:
            (需求级别, 分数)
        """
        query_lower = query.lower()

        high_count = sum(1 for kw in self.IMAGE_KEYWORDS['high'] if kw in query_lower)
        medium_count = sum(1 for kw in self.IMAGE_KEYWORDS['medium'] if kw in query_lower)
        low_count = sum(1 for kw in self.IMAGE_KEYWORDS['low'] if kw in query_lower)

        total = high_count + medium_count + low_count + 1e-6
        score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.1) / total

        if high_count >= 2 or (high_count >= 1 and medium_count >= 1):
            return 'high', min(score, 1.0)
        elif medium_count >= 2 or high_count >= 1:
            return 'medium', score
        else:
            return 'low', score

    def compute_multimodal_score(
        self,
        query: str,
        document: Dict,
        query_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算多模态融合分数

        Args:
            query: 查询文本
            document: 文档（包含 content, image_path 等）
            query_embedding: 预计算的查询嵌入

        Returns:
            (融合分数, 详细信息)
        """
        image_need, need_score = self.estimate_image_need(query)
        weights = self.WEIGHT_CONFIGS[image_need]

        if query_embedding is None:
            query_embedding = np.array(self.text_embedder.embed_text(query))

        doc_text = document.get('content', '') or document.get('problem', '') + ' ' + document.get('action', '')
        doc_embedding = np.array(self.text_embedder.embed_text(doc_text))

        text_score = float(np.dot(query_embedding, doc_embedding) /
                          (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10))

        image_path = document.get('image_path', '')
        image_score = 0.0
        summary_score = 0.0
        has_image = False

        if image_path and os.path.exists(image_path):
            has_image = True

            if image_path in self._image_cache:
                image_emb = self._image_cache[image_path]
            else:
                image_emb = self.clip_aligner.encode_image(image_path)
                if image_emb is not None:
                    self._image_cache[image_path] = image_emb

            if image_emb is not None:
                query_clip_emb = self.clip_aligner.encode_text(query)
                image_score = float(np.dot(image_emb, query_clip_emb))

            if self.vlm_summarizer is not None:
                if image_path in self._summary_cache:
                    summary = self._summary_cache[image_path]
                else:
                    summary = self.vlm_summarizer.generate_summary(image_path, query)
                    self._summary_cache[image_path] = summary

                if summary:
                    summary_emb = np.array(self.text_embedder.embed_text(summary))
                    summary_score = float(np.dot(query_embedding, summary_emb) /
                                        (np.linalg.norm(query_embedding) * np.linalg.norm(summary_emb) + 1e-10))

        if has_image:
            fusion_score = (
                weights['text'] * text_score +
                weights['image'] * image_score +
                weights['summary'] * summary_score
            )
        else:
            fusion_score = text_score

        details = {
            'text_score': text_score,
            'image_score': image_score,
            'summary_score': summary_score,
            'fusion_score': fusion_score,
            'image_need': image_need,
            'need_score': need_score,
            'has_image': has_image,
            'weights': weights
        }

        return fusion_score, details

    def rerank_with_multimodal(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        使用多模态信息重排序候选文档

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档列表
        """
        if not candidates:
            return []

        query_embedding = np.array(self.text_embedder.embed_text(query))

        scored_candidates = []
        for doc in candidates:
            fusion_score, details = self.compute_multimodal_score(
                query, doc, query_embedding
            )

            new_doc = doc.copy()
            new_doc['multimodal_score'] = fusion_score
            new_doc['original_score'] = doc.get('score', 0.0)
            new_doc['score'] = fusion_score
            new_doc['multimodal_details'] = details
            scored_candidates.append(new_doc)

        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates[:top_k]

    def clear_cache(self):
        """清除缓存"""
        self._image_cache.clear()
        self._summary_cache.clear()


class MultimodalIndexBuilder:
    """
    多模态索引构建器

    预计算图像嵌入和摘要，加速检索
    """

    def __init__(
        self,
        clip_aligner: CLIPAligner,
        vlm_summarizer: Optional[QwenVLSummarizer] = None
    ):
        self.clip_aligner = clip_aligner
        self.vlm_summarizer = vlm_summarizer

    def build_image_index(
        self,
        documents: List[Dict],
        output_path: str,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        构建图像嵌入索引

        Args:
            documents: 文档列表
            output_path: 输出路径
            batch_size: 批次大小

        Returns:
            索引统计信息
        """
        image_paths = []
        doc_ids = []

        for doc in documents:
            image_path = doc.get('image_path', '')
            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
                doc_ids.append(doc.get('doc_id', ''))

        logger.info(f"Found {len(image_paths)} images to index")

        embeddings = self.clip_aligner.batch_encode_images(image_paths, batch_size)

        index_data = {
            'embeddings': {k: v.tolist() for k, v in embeddings.items()},
            'doc_mapping': dict(zip(image_paths, doc_ids)),
            'dimension': 768,
            'total_images': len(embeddings)
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f)

        logger.info(f"Image index saved to {output_path}")

        return {
            'total_documents': len(documents),
            'documents_with_images': len(image_paths),
            'indexed_images': len(embeddings),
            'coverage': len(embeddings) / len(documents) if documents else 0
        }

    def build_summary_index(
        self,
        documents: List[Dict],
        output_path: str,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        构建图像摘要索引

        Args:
            documents: 文档列表
            output_path: 输出路径
            batch_size: 批次大小

        Returns:
            索引统计信息
        """
        if self.vlm_summarizer is None:
            logger.warning("VLM summarizer not available, skipping summary index")
            return {'error': 'VLM summarizer not available'}

        summaries = {}

        image_paths = []
        contexts = []

        for doc in documents:
            image_path = doc.get('image_path', '')
            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
                context = doc.get('problem', '') or doc.get('content', '')[:200]
                contexts.append(context)

        logger.info(f"Generating summaries for {len(image_paths)} images")

        generated_summaries = self.vlm_summarizer.batch_summarize(
            image_paths, contexts, batch_size
        )

        for path, summary in zip(image_paths, generated_summaries):
            summaries[path] = summary

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary index saved to {output_path}")

        non_empty = sum(1 for s in summaries.values() if s)

        return {
            'total_images': len(image_paths),
            'generated_summaries': non_empty,
            'empty_summaries': len(summaries) - non_empty
        }

    def load_image_index(self, index_path: str) -> Dict[str, np.ndarray]:
        """加载图像嵌入索引"""
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        embeddings = {
            k: np.array(v, dtype=np.float32)
            for k, v in index_data['embeddings'].items()
        }

        return embeddings

    def load_summary_index(self, index_path: str) -> Dict[str, str]:
        """加载摘要索引"""
        with open(index_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        return summaries


def run_multimodal_experiment(
    data_path: str,
    output_dir: str,
    text_embedder,
    use_vlm: bool = True,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    运行多模态增强实验

    Args:
        data_path: 数据路径
        output_dir: 输出目录
        text_embedder: 文本嵌入器
        use_vlm: 是否使用 VLM 摘要
        device: 设备

    Returns:
        实验结果
    """
    logger.info("Initializing multimodal components...")

    clip_aligner = CLIPAligner(device=device)

    vlm_summarizer = None
    if use_vlm:
        try:
            vlm_summarizer = QwenVLSummarizer(device=device, use_4bit=True)
        except Exception as e:
            logger.warning(f"Failed to load VLM, continuing without it: {e}")

    fusion = AdaptiveMultimodalFusion(
        text_embedder=text_embedder,
        clip_aligner=clip_aligner,
        vlm_summarizer=vlm_summarizer
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'config': {
            'use_vlm': use_vlm and vlm_summarizer is not None,
            'device': device
        },
        'metrics': {}
    }

    with open(output_path / 'multimodal_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Multimodal experiment results saved to {output_dir}")

    return results
