"""
配置管理模块
支持YAML配置文件加载和环境变量覆盖
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dimension: int = 3072
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class HMRConfig:
    """层次化多模态检索配置"""
    enabled: bool = True
    top_n_docs: int = 3
    top_m_sections: int = 5
    top_k_elements: int = 4
    summary_max_length: int = 500


@dataclass
class EntityConfig:
    """实体抽取配置"""
    enabled: bool = True
    storage_type: str = "json"
    storage_path: str = "./data/entity_store"
    confidence_threshold: float = 0.7
    max_entities_per_chunk: int = 10
    extract_relations: bool = True


@dataclass
class AMFConfig:
    """自适应模态融合配置"""
    enabled: bool = True
    confidence_threshold: float = 0.6
    default_preference: str = "balanced"
    reflection_enabled: bool = True
    reranker_method: str = "llm"


@dataclass
class MoREConfig:
    """MoRE (Mixture-of-Retrieval-Experts) 配置"""
    enabled: bool = True
    mode: str = "learned"           # learned, theory, uniform
    n_experts: int = 4
    query_dim: int = 3072
    hidden_dim: int = 256
    dropout: float = 0.1
    temperature: float = 1.0
    top_k_gate: int = 0             # 0=dense, >0=sparse
    lambda_balance: float = 0.1
    lambda_spec: float = 0.01
    lambda_entropy: float = 0.05
    checkpoint_path: Optional[str] = None


@dataclass
class TAGConfig:
    """可追溯答案生成配置"""
    enabled: bool = True
    citation_format: str = "[Source N]"
    require_evidence: bool = True
    verification_enabled: bool = True
    verification_method: str = "llm"
    confidence_threshold: float = 0.7


class Config:
    """
    统一配置管理器
    支持从YAML文件加载配置，并允许环境变量覆盖
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._config_path = config_path

        # 加载默认配置
        default_config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if default_config_path.exists():
            self._load_yaml(default_config_path)

        # 加载指定配置（覆盖默认）
        if config_path and Path(config_path).exists():
            self._load_yaml(config_path)

        # 环境变量覆盖
        self._load_env_overrides()

    def _load_yaml(self, path: Path) -> None:
        """加载YAML配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self._merge_config(config)

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """递归合并配置"""
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(self._config[key], value)
            else:
                self._config[key] = value

    def _merge_config_recursive(self, base: Dict, override: Dict) -> None:
        """递归合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(base[key], value)
            else:
                base[key] = value

    def _load_env_overrides(self) -> None:
        """从环境变量加载覆盖配置"""
        env_mappings = {
            "OPENAI_API_KEY": ("llm", "api_key"),
            "OPENAI_API_BASE": ("llm", "api_base"),
            "DASHSCOPE_API_KEY": ("llm", "api_key"),
            "DASHSCOPE_API_BASE": ("llm", "api_base"),
            "LLM_MODEL": ("llm", "model"),
            "EMBEDDING_MODEL": ("embeddings", "text", "model"),
            "EMBEDDING_API_BASE": ("embeddings", "text", "api_base"),
            "NEO4J_URI": ("entity", "neo4j", "uri"),
            "NEO4J_USER": ("entity", "neo4j", "user"),
            "NEO4J_PASSWORD": ("entity", "neo4j", "password"),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)

    def _set_nested(self, path: tuple, value: Any) -> None:
        """设置嵌套配置值"""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套路径"""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def __getitem__(self, key: str) -> Any:
        """字典式访问"""
        return self._config.get(key)

    @property
    def llm(self) -> LLMConfig:
        """获取LLM配置"""
        cfg = self.get("llm", default={})
        return LLMConfig(
            provider=cfg.get("provider", "openai"),
            model=cfg.get("model", "gpt-4o"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 4096),
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        )

    @property
    def multimodal_llm(self) -> LLMConfig:
        """获取多模态LLM配置"""
        cfg = self.get("multimodal_llm", default={})
        return LLMConfig(
            provider=cfg.get("provider", "openai"),
            model=cfg.get("model", "gpt-4o"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 2048),
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        )

    @property
    def text_embedding(self) -> EmbeddingConfig:
        """获取文本嵌入配置"""
        cfg = self.get("embeddings", "text", default={})
        return EmbeddingConfig(
            provider=cfg.get("provider", "openai"),
            model=cfg.get("model", "text-embedding-3-large"),
            dimension=cfg.get("dimension", 3072),
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )

    @property
    def image_embedding(self) -> EmbeddingConfig:
        """获取图像嵌入配置"""
        cfg = self.get("embeddings", "image", default={})
        return EmbeddingConfig(
            provider=cfg.get("provider", "clip"),
            model=cfg.get("model", "clip-vit-large-patch14"),
            dimension=cfg.get("dimension", 768),
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key")
        )

    @property
    def hmr(self) -> HMRConfig:
        """获取HMR配置"""
        cfg = self.get("hmr", default={})
        doc_cfg = cfg.get("document_level", {})
        sec_cfg = cfg.get("section_level", {})
        elem_cfg = cfg.get("element_level", {})
        return HMRConfig(
            enabled=cfg.get("enabled", True),
            top_n_docs=doc_cfg.get("top_n", 3),
            top_m_sections=sec_cfg.get("top_m", 5),
            top_k_elements=elem_cfg.get("top_k", 4),
            summary_max_length=doc_cfg.get("summary_max_length", 500)
        )

    @property
    def entity(self) -> EntityConfig:
        """获取实体配置"""
        cfg = self.get("entity", default={})
        storage_cfg = cfg.get("storage", {})
        extraction_cfg = cfg.get("extraction", {})
        return EntityConfig(
            enabled=cfg.get("enabled", True),
            storage_type=storage_cfg.get("type", "json"),
            storage_path=storage_cfg.get("path", "./data/entity_store"),
            confidence_threshold=extraction_cfg.get("confidence_threshold", 0.7),
            max_entities_per_chunk=extraction_cfg.get("max_entities_per_chunk", 10),
            extract_relations=extraction_cfg.get("extract_relations", True)
        )

    @property
    def amf(self) -> AMFConfig:
        """获取AMF配置"""
        cfg = self.get("amf", default={})
        qc_cfg = cfg.get("query_classifier", {})
        ref_cfg = cfg.get("reflection", {})
        rerank_cfg = cfg.get("reranker", {})
        return AMFConfig(
            enabled=cfg.get("enabled", True),
            confidence_threshold=qc_cfg.get("confidence_threshold", 0.6),
            default_preference=qc_cfg.get("default_preference", "balanced"),
            reflection_enabled=ref_cfg.get("enabled", True),
            reranker_method=rerank_cfg.get("method", "llm")
        )

    @property
    def more(self) -> MoREConfig:
        """获取MoRE配置"""
        cfg = self.get("more", default={})
        gating_cfg = cfg.get("gating", {})
        training_cfg = cfg.get("training", {})
        return MoREConfig(
            enabled=cfg.get("enabled", True),
            mode=cfg.get("mode", "learned"),
            n_experts=gating_cfg.get("n_experts", 4),
            query_dim=gating_cfg.get("query_dim", 3072),
            hidden_dim=gating_cfg.get("hidden_dim", 256),
            dropout=gating_cfg.get("dropout", 0.1),
            temperature=gating_cfg.get("temperature", 1.0),
            top_k_gate=gating_cfg.get("top_k", 0),
            lambda_balance=training_cfg.get("lambda_balance", 0.1),
            lambda_spec=training_cfg.get("lambda_spec", 0.01),
            lambda_entropy=training_cfg.get("lambda_entropy", 0.05),
            checkpoint_path=cfg.get("checkpoint_path")
        )

    @property
    def tag(self) -> TAGConfig:
        """获取TAG配置"""
        cfg = self.get("tag", default={})
        citation_cfg = cfg.get("citation", {})
        verify_cfg = cfg.get("verification", {})
        return TAGConfig(
            enabled=cfg.get("enabled", True),
            citation_format=citation_cfg.get("format", "[Source N]"),
            require_evidence=citation_cfg.get("require_evidence", True),
            verification_enabled=verify_cfg.get("enabled", True),
            verification_method=verify_cfg.get("method", "llm"),
            confidence_threshold=cfg.get("confidence_threshold", 0.7)
        )

    @property
    def paths(self) -> Dict[str, str]:
        """获取路径配置"""
        return self.get("paths", default={
            "data_dir": "./data",
            "raw_dir": "./data/raw",
            "processed_dir": "./data/processed",
            "embeddings_dir": "./data/embeddings",
            "entity_store_dir": "./data/entity_store",
            "results_dir": "./experiments/results",
            "logs_dir": "./experiments/logs"
        })

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return self._config.copy()

    def save(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


# 全局配置实例
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    return _global_config
