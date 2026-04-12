#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
export UV_LINK_MODE=copy

echo "=== Step 1: Install doc-to-lora (tolerating network failures) ==="
cd "$(dirname "$0")/../.."

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv self update
uv venv --python 3.10 --seed
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124
uv sync
uv pip install tokenizers==0.21.0

# flash-attn / flashinfer: optional, skip on network failure
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || echo "WARN: flash-attn install failed, skipping (not required for diagnostic)"
uv pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6 || echo "WARN: flashinfer install failed, skipping (not required for diagnostic)"

# download datasets used by install.sh (via mirror)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad
uv run data/build_drop_compact.py
uv run data/build_pwc_compact.py
uv run data/build_ropes_compact.py
uv run data/build_squad_compact.py

echo "=== Step 2: Download doc-to-lora checkpoint ==="
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download SakanaAI/doc-to-lora \
    --local-dir trained_d2l

echo "=== Step 3: Download PERMA dataset ==="
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download ustclsc/PERMA \
    --repo-type dataset \
    --local-dir experiments/perma/data \
    --resume-download

echo "=== Step 4: Run smoke test (1 user) ==="
PERMA_DATA_ROOT=experiments/perma/data \
uv run experiments/perma/diagnostic_eval.py \
    --checkpoint trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
    --mode oracle \
    --max_users 1

echo "=== Step 5: Run full diagnostic (all 3 modes, 2 users) ==="
PERMA_DATA_ROOT=experiments/perma/data \
uv run experiments/perma/diagnostic_eval.py \
    --checkpoint trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
    --mode all \
    --max_users 2

echo "=== Done! Check results in experiments/perma/results/ ==="
