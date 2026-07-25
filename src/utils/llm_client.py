"""
LLM客户端封装
支持OpenAI、Anthropic等多种LLM提供商
"""

import os
import json
import base64
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .config import LLMConfig, get_config
from .logger import LoggerMixin


class BaseLLMClient(ABC, LoggerMixin):
    """LLM客户端基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成文本响应"""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成结构化JSON响应"""
        pass

    @abstractmethod
    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, Path],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """图像+文本多模态生成"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI兼容客户端 (支持OpenAI/DashScope/本地模型)"""
        try:
            from openai import OpenAI

            # 支持多种API密钥环境变量
            api_key = (
                self.config.api_key or
                os.environ.get("OPENAI_API_KEY") or
                os.environ.get("DASHSCOPE_API_KEY") or
                os.environ.get("LLM_API_KEY")
            )

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.api_base
            )
            self.logger.info(f"LLM client initialized: {self.config.model} @ {self.config.api_base or 'default'}")
        except ImportError:
            self.logger.error("OpenAI package not installed. Run: pip install openai")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成文本响应"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            **kwargs
        )

        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成结构化JSON响应"""
        # 构建系统提示
        json_system_prompt = system_prompt or ""
        json_system_prompt += "\n\nYou must respond with valid JSON only. No additional text."

        if schema:
            json_system_prompt += f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"

        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            **kwargs
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {content}")
            return {}

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, Path],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """图像+文本多模态生成"""
        # 处理图像
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                # 根据文件扩展名确定MIME类型
                ext = image_path.suffix.lower()
                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }.get(ext, "image/png")
                image_url = f"data:{mime_type};base64,{image_data}"
            else:
                # 假设是URL
                image_url = str(image)
        elif isinstance(image, bytes):
            image_data = base64.b64encode(image).decode("utf-8")
            image_url = f"data:image/png;base64,{image_data}"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )

        return response.choices[0].message.content

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, bytes, Path]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """多图像+文本多模态生成"""
        content = [{"type": "text", "text": prompt}]

        for image in images:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    ext = image_path.suffix.lower()
                    mime_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg"
                    }.get(ext, "image/png")
                    image_url = f"data:{mime_type};base64,{image_data}"
                else:
                    image_url = str(image)
            elif isinstance(image, bytes):
                image_data = base64.b64encode(image).decode("utf-8")
                image_url = f"data:image/png;base64,{image_data}"
            else:
                continue

            content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )

        return response.choices[0].message.content


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API客户端"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化Anthropic客户端"""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            self.logger.error("Anthropic package not installed. Run: pip install anthropic")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成文本响应"""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.config.temperature,
            **kwargs
        )

        return response.content[0].text

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成结构化JSON响应"""
        json_prompt = prompt + "\n\nRespond with valid JSON only."
        if schema:
            json_prompt += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"

        response = self.generate(json_prompt, system_prompt, **kwargs)

        try:
            # 尝试提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return {}

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, Path],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """图像+文本多模态生成"""
        # 处理图像
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                ext = image_path.suffix.lower()
                media_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }.get(ext, "image/png")
            else:
                raise ValueError(f"Image file not found: {image_path}")
        elif isinstance(image, bytes):
            image_data = base64.b64encode(image).decode("utf-8")
            media_type = "image/png"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt or "",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            **kwargs
        )

        return response.content[0].text


class LocalTransformersClient(BaseLLMClient):
    """本地 Transformers 模型客户端（支持 Qwen 等）"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._model = None
        self._tokenizer = None
        self._device = None
        self._init_model()

    def _init_model(self):
        """初始化本地模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.model

        # 检查常用路径
        if not os.path.exists(model_path):
            # 尝试 models 目录
            alt_paths = [
                f"./models/Qwen/{model_path}",
                f"./models/Qwen/Qwen2___5-1___5B-Instruct",
                f"./models/Qwen/Qwen2.5-1.5B-Instruct",
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Loading local model: {model_path} on {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True
        )

        if self._device == "cpu":
            self._model = self._model.to(self._device)

        self.logger.info(f"Local model loaded successfully on {self._device}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成文本响应"""
        import torch

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 使用 chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.max_tokens or 512,
                temperature=temperature or self.config.temperature or 0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # 解码，去掉输入部分
        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成结构化JSON响应"""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no explanation."
        if schema:
            json_prompt += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"

        response = self.generate(json_prompt, system_prompt, **kwargs)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])

            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            return {}
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, Path],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """图像+文本（本地模型暂不支持多模态，返回纯文本提示）"""
        self.logger.warning("Local model does not support image input, using text-only")
        return self.generate(prompt, system_prompt, **kwargs)


class LocalVLClient(BaseLLMClient):
    """本地 Qwen2-VL 多模态模型客户端"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._model = None
        self._processor = None
        self._device = None
        self._init_model()

    def _init_model(self):
        """初始化 Qwen2-VL 多模态模型"""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_path = self.config.model

        # 检查路径是否存在
        if not os.path.exists(model_path):
            alt_paths = [
                f"./models/Qwen/{model_path}",
                f"./models/Qwen/Qwen2-VL-7B-Instruct",
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading Qwen2-VL model: {model_path} on {self._device}")

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True
        )

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self._device == "cpu":
            self._model = self._model.to(self._device)

        self.logger.info(f"Qwen2-VL model loaded successfully on {self._device}")

    def _generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """从消息列表生成响应（内部方法）"""
        import torch

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 检查是否有视觉输入
        has_vision = any(
            isinstance(msg.get("content"), list) and
            any(c.get("type") == "image" for c in msg["content"] if isinstance(c, dict))
            for msg in messages
        )

        if has_vision:
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                # 手动处理图像输入
                image_inputs = self._extract_images_from_messages(messages)
                video_inputs = None

            inputs = self._processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self._processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )

        inputs = inputs.to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens or self.config.max_tokens or 2048,
        }
        temp = temperature or self.config.temperature or 0.1
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # 裁剪输入部分
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    def _extract_images_from_messages(self, messages: List[Dict]) -> List:
        """从消息中提取图像（qwen_vl_utils 不可用时的后备方案）"""
        from PIL import Image as PILImage

        images = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                img_source = item.get("image", "")
                if img_source.startswith("file://"):
                    img_path = img_source[7:]
                    if os.path.exists(img_path):
                        images.append(PILImage.open(img_path).convert("RGB"))
                elif os.path.exists(img_source):
                    images.append(PILImage.open(img_source).convert("RGB"))
        return images if images else None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成纯文本响应"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        return self._generate_from_messages(messages, max_tokens, temperature)

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成结构化JSON响应"""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no explanation."
        if schema:
            json_prompt += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"

        response = self.generate(json_prompt, system_prompt, **kwargs)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            return {}
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, Path],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """图像+文本多模态生成（Qwen2-VL 原生支持）"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建图像路径
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                img_uri = f"file://{image_path.resolve()}"
            else:
                img_uri = str(image)
        elif isinstance(image, bytes):
            # 保存临时文件
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(image)
            tmp.close()
            img_uri = f"file://{tmp.name}"
        else:
            self.logger.warning(f"Unsupported image type: {type(image)}, falling back to text-only")
            return self.generate(prompt, system_prompt, **kwargs)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_uri},
                {"type": "text", "text": prompt}
            ]
        })

        return self._generate_from_messages(messages)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, bytes, Path]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """多图像+文本多模态生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        for image in images:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if image_path.exists():
                    content.append({"type": "image", "image": f"file://{image_path.resolve()}"})
                else:
                    content.append({"type": "image", "image": str(image)})
            elif isinstance(image, bytes):
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(image)
                tmp.close()
                content.append({"type": "image", "image": f"file://{tmp.name}"})

        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        return self._generate_from_messages(messages)


class LLMClient:
    """
    统一LLM客户端工厂
    根据配置自动选择合适的客户端
    """

    _instances: Dict[str, BaseLLMClient] = {}

    @classmethod
    def get_client(
        cls,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None
    ) -> BaseLLMClient:
        """
        获取LLM客户端实例

        Args:
            config: LLM配置
            provider: 提供商名称 (openai, anthropic, local)

        Returns:
            LLM客户端实例
        """
        if config is None:
            config = get_config().llm

        provider = provider or config.provider

        # 缓存key
        cache_key = f"{provider}_{config.model}"

        if cache_key not in cls._instances:
            if provider == "openai":
                cls._instances[cache_key] = OpenAIClient(config)
            elif provider == "anthropic":
                cls._instances[cache_key] = AnthropicClient(config)
            elif provider == "local":
                cls._instances[cache_key] = LocalTransformersClient(config)
            elif provider == "local_vl":
                cls._instances[cache_key] = LocalVLClient(config)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        """清除客户端缓存"""
        cls._instances.clear()
