#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"
DATA_DIR="data/raw_datasets/perma"

# Step 1: 生成训练数据
echo "=== Preparing training data ==="
PERMA_DATA_ROOT=experiments/perma/data \
  uv run experiments/perma/prepare_train_data.py \
      --test_user 334 \
      --output_dir "$DATA_DIR"

# Step 2: 微调 (单卡)
echo "=== Fine-tuning ==="
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  configs/perma/finetune_mistral.yaml \
  --from_pretrained_checkpoint=$CKPT \
  --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2 \
  --target_modules=down_proj --lora_r=8 \
  --eval_strategy=no \
  --max_qas_len=2048 --max_qas_per_sample=1 \
  --per_rank_gen=True --per_layer_processing=True \
  --gen_lora_l1_reg_coef=0.01 \
  --max_steps=2000 --save_steps=500 \
  --learning_rate=1e-5 --warmup_steps=50 \
  --use_per_ctx_average_loss=True \
  --use_kl_loss=False

echo "=== Done ==="
