#!/usr/bin/env python3
"""
HEA-MRAG 模型自动下载脚本
自动检测网络环境，下载所需的AI模型
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple

def check_network_environment() -> str:
    """检测网络环境，选择最佳下载源"""
    print("🌐 检测网络环境...")

    # 测试ModelScope连接（国内）
    try:
        response = requests.get("https://modelscope.cn", timeout=5)
        if response.status_code == 200:
            print("  ✅ 检测到国内网络环境，将使用 ModelScope")
            return "modelscope"
    except:
        pass

    # 测试Hugging Face连接（国外）
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        if response.status_code == 200:
            print("  ✅ 检测到国外网络环境，将使用 Hugging Face")
            return "huggingface"
    except:
        pass

    # 测试HF镜像连接
    try:
        response = requests.get("https://hf-mirror.com", timeout=5)
        if response.status_code == 200:
            print("  ✅ 将使用 Hugging Face 镜像")
            return "hf-mirror"
    except:
        pass

    print("  ⚠️  网络连接检测失败，将尝试使用 ModelScope")
    return "modelscope"

def get_model_list() -> List[Dict]:
    """获取需要下载的模型列表"""
    return [
        {
            "name": "Qwen2-VL-7B-Instruct",
            "size": "18.6GB",
            "priority": 1,
            "required": True,
            "description": "视觉语言模型（多模态理解）",
            "modelscope_id": "qwen/Qwen2-VL-7B-Instruct",
            "huggingface_id": "Qwen/Qwen2-VL-7B-Instruct"
        },
        {
            "name": "Qwen2-7B-Instruct",
            "size": "14.3GB",
            "priority": 1,
            "required": True,
            "description": "文本生成模型（答案生成）",
            "modelscope_id": "qwen/Qwen2-7B-Instruct",
            "huggingface_id": "Qwen/Qwen2-7B-Instruct"
        },
        {
            "name": "bge-small-zh-v1.5",
            "size": "400MB",
            "priority": 1,
            "required": True,
            "description": "中文文本嵌入模型",
            "modelscope_id": "AI-ModelScope/bge-small-zh-v1.5",
            "huggingface_id": "BAAI/bge-small-zh-v1.5"
        },
        {
            "name": "CLIP-ViT-Large",
            "size": "6.4GB",
            "priority": 2,
            "required": False,
            "description": "图像编码器（图像检索）",
            "modelscope_id": "AI-ModelScope/clip-vit-large-patch14",
            "huggingface_id": "openai/clip-vit-large-patch14"
        },
        {
            "name": "Qwen2-1.5B-Instruct",
            "size": "2.9GB",
            "priority": 3,
            "required": False,
            "description": "轻量级文本模型（可选）",
            "modelscope_id": "qwen/Qwen2-1.5B-Instruct",
            "huggingface_id": "Qwen/Qwen2-1.5B-Instruct"
        }
    ]

def check_disk_space(required_gb: float) -> bool:
    """检查磁盘空间是否足够"""
    import shutil

    free_bytes = shutil.disk_usage(".").free
    free_gb = free_bytes / (1024**3)

    print(f"💾 磁盘空间检查:")
    print(f"  可用空间: {free_gb:.1f} GB")
    print(f"  需要空间: {required_gb:.1f} GB")

    if free_gb < required_gb:
        print(f"  ❌ 磁盘空间不足！")
        return False
    else:
        print(f"  ✅ 磁盘空间充足")
        return True

def install_dependencies(source: str):
    """安装下载依赖"""
    print("📦 安装下载依赖...")

    if source == "modelscope":
        os.system("pip install modelscope -q")
    else:
        os.system("pip install huggingface_hub -q")

    print("  ✅ 依赖安装完成")

def download_model_modelscope(model_info: Dict, models_dir: Path) -> bool:
    """使用ModelScope下载模型"""
    try:
        from modelscope import snapshot_download

        model_name = model_info["name"]
        model_id = model_info["modelscope_id"]
        local_dir = models_dir / model_name

        print(f"  📥 从 ModelScope 下载 {model_name}...")

        snapshot_download(
            model_id,
            cache_dir=str(models_dir),
            local_dir=str(local_dir),
        )

        return True

    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False

def download_model_huggingface(model_info: Dict, models_dir: Path, use_mirror: bool = False) -> bool:
    """使用Hugging Face下载模型"""
    try:
        from huggingface_hub import snapshot_download

        model_name = model_info["name"]
        model_id = model_info["huggingface_id"]
        local_dir = models_dir / model_name

        # 设置镜像环境变量
        if use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"  📥 从 HF镜像 下载 {model_name}...")
        else:
            print(f"  📥 从 Hugging Face 下载 {model_name}...")

        snapshot_download(
            model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )

        return True

    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False

def verify_model(model_dir: Path) -> bool:
    """验证模型文件完整性"""
    if not model_dir.exists():
        return False

    # 检查关键文件
    required_files = ["config.json"]
    optional_files = ["pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"]

    # 必须有config.json
    if not (model_dir / "config.json").exists():
        return False

    # 必须有模型权重文件（任一格式）
    has_weights = any((model_dir / f).exists() for f in optional_files)
    if not has_weights:
        # 检查分片模型文件
        safetensors_files = list(model_dir.glob("model-*.safetensors"))
        bin_files = list(model_dir.glob("pytorch_model-*.bin"))
        has_weights = len(safetensors_files) > 0 or len(bin_files) > 0

    return has_weights

def update_config_file(models_dir: Path, downloaded_models: List[str]):
    """更新配置文件中的模型路径"""
    config_file = Path("configs/local.yaml")

    if not config_file.exists():
        # 创建基础配置文件
        config_content = f"""# HEA-MRAG 本地配置文件
# 自动生成于模型下载完成后

_base_: "default.yaml"

# 模型路径配置
models:
  model_dir: "{models_dir.absolute()}"
"""

        # 根据下载的模型添加具体路径
        if "bge-small-zh-v1.5" in downloaded_models:
            config_content += f'  text_encoder: "{models_dir.absolute()}/bge-small-zh-v1.5"\n'

        if "Qwen2-7B-Instruct" in downloaded_models:
            config_content += f'  llm: "{models_dir.absolute()}/Qwen2-7B-Instruct"\n'

        if "Qwen2-VL-7B-Instruct" in downloaded_models:
            config_content += f'  multimodal_llm: "{models_dir.absolute()}/Qwen2-VL-7B-Instruct"\n'

        if "CLIP-ViT-Large" in downloaded_models:
            config_content += f'  image_encoder: "{models_dir.absolute()}/CLIP-ViT-Large"\n'

        config_content += "\n# 计算资源配置\ncompute:\n  device: \"auto\"  # 自动检测GPU/CPU\n  batch_size: 16\n  max_length: 512\n"

        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)

        print(f"  ✅ 已创建配置文件: {config_file}")
    else:
        print(f"  ℹ️  配置文件已存在: {config_file}")

def main():
    """主函数"""
    print("🤖 HEA-MRAG 模型自动下载工具")
    print("=" * 50)

    # 1. 检测网络环境
    source = check_network_environment()

    # 2. 获取模型列表
    models = get_model_list()

    # 3. 显示模型信息
    print("\n📋 模型下载清单:")
    total_size_gb = 0
    required_size_gb = 0

    for i, model in enumerate(models, 1):
        status = "必需" if model["required"] else "可选"
        priority = "⭐" * model["priority"]
        print(f"  {i}. {model['name']} ({model['size']}) - {model['description']} [{status}] {priority}")

        # 计算大小（简单估算）
        size_str = model["size"]
        if "GB" in size_str:
            size_gb = float(size_str.replace("GB", ""))
        elif "MB" in size_str:
            size_gb = float(size_str.replace("MB", "")) / 1024
        else:
            size_gb = 0

        total_size_gb += size_gb
        if model["required"]:
            required_size_gb += size_gb

    print(f"\n📊 存储需求:")
    print(f"  必需模型: {required_size_gb:.1f} GB")
    print(f"  全部模型: {total_size_gb:.1f} GB")

    # 4. 选择下载模式
    print(f"\n❓ 请选择下载模式:")
    print(f"  1. 仅下载必需模型 ({required_size_gb:.1f} GB)")
    print(f"  2. 下载全部模型 ({total_size_gb:.1f} GB)")
    print(f"  3. 自定义选择")

    try:
        choice = input("请输入选择 (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # 默认选择必需模型

    # 确定要下载的模型
    models_to_download = []

    if choice == "1":
        models_to_download = [m for m in models if m["required"]]
        target_size = required_size_gb
    elif choice == "2":
        models_to_download = models
        target_size = total_size_gb
    elif choice == "3":
        print("\n请选择要下载的模型 (输入序号，用空格分隔):")
        try:
            indices = input("模型序号: ").strip().split()
            for idx in indices:
                if idx.isdigit() and 1 <= int(idx) <= len(models):
                    models_to_download.append(models[int(idx)-1])
        except (EOFError, KeyboardInterrupt):
            models_to_download = [m for m in models if m["required"]]

        target_size = sum(float(m["size"].replace("GB", "").replace("MB", "")) / (1024 if "MB" in m["size"] else 1)
                         for m in models_to_download)
    else:
        print("无效选择，将下载必需模型")
        models_to_download = [m for m in models if m["required"]]
        target_size = required_size_gb

    if not models_to_download:
        print("❌ 没有选择任何模型")
        return

    # 5. 检查磁盘空间
    if not check_disk_space(target_size + 5):  # 额外5GB缓冲
        print("\n💡 建议:")
        print("  - 清理磁盘空间")
        print("  - 选择仅下载必需模型")
        print("  - 将模型下载到外部存储")
        return

    # 6. 创建模型目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 7. 安装依赖
    install_dependencies(source)

    # 8. 开始下载
    print(f"\n🚀 开始下载 {len(models_to_download)} 个模型...")

    downloaded_models = []
    failed_models = []

    for i, model in enumerate(models_to_download, 1):
        model_name = model["name"]
        model_dir = models_dir / model_name

        print(f"\n[{i}/{len(models_to_download)}] {model_name} ({model['size']})")

        # 检查是否已存在
        if verify_model(model_dir):
            print(f"  ✅ 模型已存在，跳过下载")
            downloaded_models.append(model_name)
            continue

        # 下载模型
        success = False

        if source == "modelscope":
            success = download_model_modelscope(model, models_dir)
        elif source == "hf-mirror":
            success = download_model_huggingface(model, models_dir, use_mirror=True)
        else:
            success = download_model_huggingface(model, models_dir, use_mirror=False)

        # 验证下载结果
        if success and verify_model(model_dir):
            print(f"  ✅ {model_name} 下载完成")
            downloaded_models.append(model_name)
        else:
            print(f"  ❌ {model_name} 下载失败")
            failed_models.append(model_name)

    # 9. 更新配置文件
    if downloaded_models:
        print(f"\n⚙️  更新配置文件...")
        update_config_file(models_dir, downloaded_models)

    # 10. 显示结果
    print(f"\n🎉 下载完成!")
    print(f"  成功: {len(downloaded_models)} 个模型")
    print(f"  失败: {len(failed_models)} 个模型")

    if downloaded_models:
        print(f"\n✅ 已下载的模型:")
        for model_name in downloaded_models:
            print(f"  - {model_name}")

    if failed_models:
        print(f"\n❌ 下载失败的模型:")
        for model_name in failed_models:
            print(f"  - {model_name}")
        print(f"\n💡 可以稍后重新运行此脚本继续下载失败的模型")

    # 11. 下一步提示
    if downloaded_models:
        print(f"\n🚀 下一步:")
        print(f"  1. 运行系统测试: python interactive_retrieval.py")
        print(f"  2. 启动Web演示: python demo/app.py")
        print(f"  3. 查看完整指南: 交付文档_HEA-MRAG_完整指南.md")

if __name__ == "__main__":
    main()