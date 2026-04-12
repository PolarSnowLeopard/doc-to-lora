#!/bin/bash
set -e

echo "=== Step 1: Install doc-to-lora ==="
cd "$(dirname "$0")/../.."
./install.sh

echo "=== Step 2: Download doc-to-lora checkpoint ==="
uv run huggingface-cli login
uv run huggingface-cli download SakanaAI/doc-to-lora \
    --local-dir trained_d2l --include "*/"

echo "=== Step 3: Download PERMA dataset ==="
uv run huggingface-cli download ustclsc/PERMA \
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
