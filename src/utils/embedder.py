"""
嵌入模型封装
支持文本嵌入和图像嵌入（CLIP）
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path
import numpy as np

from .config import EmbeddingConfig, get_config
from .logger import LoggerMixin


class BaseEmbedder(ABC, LoggerMixin):
    """嵌入模型基类"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入维度"""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI兼容文本嵌入 (支持OpenAI/DashScope/本地模型)"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().text_embedding
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI

            # 支持多种API密钥环境变量
            api_key = (
                getattr(self.config, 'api_key', None) or
                os.environ.get("OPENAI_API_KEY") or
                os.environ.get("DASHSCOPE_API_KEY") or
                os.environ.get("EMBEDDING_API_KEY")
            )

            # 支持自定义API地址 (DashScope, 本地模型等)
            api_base = getattr(self.config, 'api_base', None)

            self._client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            self.logger.info(f"Embedding client initialized with base_url: {api_base or 'default'}")
        except ImportError:
            self.logger.error("OpenAI package not installed. Run: pip install openai")
            raise

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        response = self._client.embeddings.create(
            model=self.config.model,
            input=text
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        # DashScope 限制批量大小为 10，OpenAI 限制为 2048
        # 使用较小的批量大小以兼容两者
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self.config.model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.config.dimension


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace文本嵌入"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().text_embedding
        self._model = None
        self._tokenizer = None
        self._dimension = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.model)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            self.logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension or self.config.dimension


class CLIPEmbedder(BaseEmbedder):
    """CLIP多模态嵌入（文本+图像）"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().image_embedding
        self._model = None
        self._processor = None
        self._device = None
        self._init_model()

    def _init_model(self):
        """初始化CLIP模型"""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.config.model

            # 常用CLIP模型映射
            model_mapping = {
                "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
                "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
                "clip-vit-base-patch16": "openai/clip-vit-base-patch16"
            }

            model_id = model_mapping.get(model_name, model_name)

            # 检查本地模型路径
            import os
            local_model_path = os.environ.get('CLIP_MODEL_PATH')
            if not local_model_path:
                # 检查 ModelScope 下载路径
                modelscope_path = f"./models/AI-ModelScope/{model_name}"
                if os.path.exists(modelscope_path):
                    local_model_path = modelscope_path

            if local_model_path and os.path.exists(local_model_path):
                self.logger.info(f"Loading CLIP from local path: {local_model_path}")
                self._model = CLIPModel.from_pretrained(local_model_path).to(self._device)
                self._processor = CLIPProcessor.from_pretrained(local_model_path)
            else:
                self._model = CLIPModel.from_pretrained(model_id).to(self._device)
                self._processor = CLIPProcessor.from_pretrained(model_id)

            self.logger.info(f"CLIP model loaded: {model_id} on {self._device}")

        except ImportError:
            self.logger.error("transformers/torch not installed. Run: pip install transformers torch")
            raise

    def embed_text(self, text: str) -> List[float]:
        """嵌入文本"""
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features[0].cpu().numpy().tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        import torch

        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_embeddings.extend(text_features.cpu().numpy().tolist())

        return all_embeddings

    def embed_image(self, image: Union[str, Path, "Image"]) -> List[float]:
        """嵌入图像"""
        import torch
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features[0].cpu().numpy().tolist()

    def embed_images(self, images: List[Union[str, Path, "Image"]]) -> List[List[float]]:
        """批量嵌入图像"""
        import torch
        from PIL import Image

        batch_size = 16
        all_embeddings = []

        # 加载图像
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.extend(image_features.cpu().numpy().tolist())

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.config.dimension


class Embedder:
    """
    统一嵌入器工厂
    """

    _text_embedder: Optional[BaseEmbedder] = None
    _image_embedder: Optional[CLIPEmbedder] = None

    @classmethod
    def get_text_embedder(
        cls,
        config: Optional[EmbeddingConfig] = None,
        provider: Optional[str] = None
    ) -> BaseEmbedder:
        """获取文本嵌入器"""
        if cls._text_embedder is None:
            if config is None:
                config = get_config().text_embedding
            provider = provider or config.provider

            if provider == "openai":
                cls._text_embedder = OpenAIEmbedder(config)
            elif provider == "huggingface":
                cls._text_embedder = HuggingFaceEmbedder(config)
            elif provider == "clip":
                cls._text_embedder = CLIPEmbedder(config)
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        return cls._text_embedder

    @classmethod
    def get_image_embedder(
        cls,
        config: Optional[EmbeddingConfig] = None
    ) -> CLIPEmbedder:
        """获取图像嵌入器"""
        if cls._image_embedder is None:
            if config is None:
                config = get_config().image_embedding
            cls._image_embedder = CLIPEmbedder(config)

        return cls._image_embedder

    @classmethod
    def clear_cache(cls):
        """清除缓存"""
        cls._text_embedder = None
        cls._image_embedder = None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_cosine_similarity(query: List[float], vectors: List[List[float]]) -> List[float]:
    """批量计算余弦相似度"""
    query_arr = np.array(query)
    vectors_arr = np.array(vectors)

    # 归一化
    query_norm = query_arr / np.linalg.norm(query_arr)
    vectors_norm = vectors_arr / np.linalg.norm(vectors_arr, axis=1, keepdims=True)

    # 点积
    similarities = np.dot(vectors_norm, query_norm)
    return similarities.tolist()
