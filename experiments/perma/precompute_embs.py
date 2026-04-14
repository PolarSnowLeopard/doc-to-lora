"""
预计算 D2L Aggregator 输出（lora_emb）并保存到磁盘。

对每个 task 的每个 session 独立运行 frozen Encoder → Aggregator，
将 lora_emb [1, n_layers, n_modules, r, d_latent] 保存为 .pt 文件。

目录结构: {output_dir}/user{uid}/{task_id}_type{task_type}/
  - session_0.pt, session_1.pt, ...
  - meta.pt  (task_type, question, options, gold_label 等)

用法:
  PERMA_DATA_ROOT=experiments/perma/data \
  python experiments/perma/precompute_embs.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --output_dir experiments/perma/cached_embs \
    --max_ctx_tokens 4000
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from data_adapter import load_tasks, session_to_text, ALL_USER_IDS


def precompute(args):
    print(f"Loading D2L model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.eval()

    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    user_ids = ALL_USER_IDS if args.max_users <= 0 else ALL_USER_IDS[:args.max_users]
    tasks = load_tasks(user_ids=user_ids, noise=False, multi_domain=False)
    print(f"Loaded {len(tasks)} tasks from {len(set(t.user_id for t in tasks))} users")

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0

    for i, task in enumerate(tasks):
        task_key = f"{task.task_id}_type{task.task_type}"
        task_dir = os.path.join(args.output_dir, f"user{task.user_id}", task_key)
        os.makedirs(task_dir, exist_ok=True)

        emb_shape = None
        for s_idx, session in enumerate(task.sessions):
            cache_path = os.path.join(task_dir, f"session_{s_idx}.pt")
            if os.path.exists(cache_path) and not args.overwrite:
                if emb_shape is None:
                    emb_shape = torch.load(
                        cache_path, weights_only=True, map_location="cpu",
                    ).shape
                continue

            text = session_to_text(session)
            tokens = ctx_tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > args.max_ctx_tokens:
                tokens = tokens[:args.max_ctx_tokens]

            ctx_ids = torch.tensor([tokens], device=model.device)
            ctx_attn_mask = torch.ones_like(ctx_ids)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ctx_features = model.ctx_encoder(
                    input_ids=ctx_ids, attention_mask=ctx_attn_mask,
                )
                lora_emb, _ = model.hypernet.aggregator(
                    ctx_features, ctx_attn_mask, None,
                )

            emb_cpu = lora_emb.cpu()
            torch.save(emb_cpu, cache_path)
            if emb_shape is None:
                emb_shape = emb_cpu.shape
            saved += 1

        meta = {
            "user_id": task.user_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "n_sessions": len(task.sessions),
            "question": task.question,
            "options": task.options,
            "gold_label": task.gold_label,
            "emb_shape": list(emb_shape) if emb_shape else [],
        }
        torch.save(meta, os.path.join(task_dir, "meta.pt"))

        if (i + 1) % 50 == 0 or i == len(tasks) - 1:
            print(f"  [{i+1}/{len(tasks)}] user{task.user_id}/{task_key}: "
                  f"{len(task.sessions)} sessions, shape={emb_shape}")

    print(f"\nDone. Saved {saved} new embeddings to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/perma/cached_embs")
    parser.add_argument("--max_ctx_tokens", type=int, default=4000)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    precompute(parser.parse_args())
