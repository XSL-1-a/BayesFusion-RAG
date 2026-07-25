#!/bin/bash
# HEA-MRAG 离线环境配置脚本
# 用于在离线服务器上部署

echo "=========================================="
echo "HEA-MRAG 离线环境配置"
echo "=========================================="

# 1. 创建conda环境
echo "[1/4] 创建conda环境..."
conda create -n hea-mrag python=3.10 -y
conda activate hea-mrag

# 2. 安装PyTorch (离线包需提前下载)
echo "[2/4] 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖
echo "[3/4] 安装依赖..."
pip install -r requirements.txt

# 4. 验证安装
echo "[4/4] 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformer: OK')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers: OK')"

echo "=========================================="
echo "环境配置完成!"
echo "=========================================="
