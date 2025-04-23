#!/bin/bash
set -e

apt-get update

# System setup
apt-get install -y curl git sudo build-essential git-lfs

# Initialize git-lfs
git lfs install

# UV install with Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/SulavKhadka/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
uv add psycopg psycopg-binary pgvector sentence-transformers hf-transfer nvitop
source .venv/bin/activate

# Add HF transfer to system-wide environment
echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/environment

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start tailscaled and authenticate
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &
sleep 2
tailscale up

chsh -s $(which zsh)



# Launch vLLM inference server from verifiers/, with .venv active
CUDA_VISIBLE_DEVICES=1 python verifiers/inference/vllm_serve.py --model "Qwen/Qwen2.5-1.5B-Instruct" --max_model_len 16384  --gpu_memory_utilization 0.5 --enable_prefix_caching True
# Run training script from verifiers/, with .venv active
CUDA_VISIBLE_DEVICES=0 accelerate launch verifiers/examples/math_train.py



# uv add psycopg psycopg-binary pgvector sentence-transformers hf-transfer nvitop
