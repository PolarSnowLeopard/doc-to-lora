"""
预计算 LoCoMo 各 session 的 D2L Aggregator 输出。

目录结构: {output_dir}/{sample_id}/
  - session_0.pt, session_1.pt, ...
  - meta.pt

用法:
  uv run experiments/locomo/precompute_embs.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --data_file experiments/locomo/data/locomo10.json \
    --output_dir experiments/locomo/cached_embs
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from data_adapter import load_conversations


def precompute(args):
    print(f"Loading D2L model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.eval()

    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    conversations = load_conversations(args.data_file)
    print(f"Loaded {len(conversations)} conversations")

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0

    for conv in conversations:
        conv_dir = os.path.join(args.output_dir, conv.sample_id)
        os.makedirs(conv_dir, exist_ok=True)

        session_texts = []
        emb_shape = None

        for s_idx, session in enumerate(conv.sessions):
            cache_path = os.path.join(conv_dir, f"session_{s_idx}.pt")
            session_texts.append(session["text"])

            if os.path.exists(cache_path) and not args.overwrite:
                if emb_shape is None:
                    emb_shape = torch.load(
                        cache_path, weights_only=True, map_location="cpu",
                    ).shape
                continue

            text = session["text"]
            tokens = ctx_tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > args.max_ctx_tokens:
                tokens = tokens[:args.max_ctx_tokens]

            ctx_ids = torch.tensor([tokens], device=model.device)
            ctx_attn_mask = torch.ones_like(ctx_ids)

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
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
            "sample_id": conv.sample_id,
            "speaker_a": conv.speaker_a,
            "speaker_b": conv.speaker_b,
            "n_sessions": len(conv.sessions),
            "session_texts": session_texts,
            "qa": conv.qa,
            "emb_shape": list(emb_shape) if emb_shape else [],
        }
        torch.save(meta, os.path.join(conv_dir, "meta.pt"))

        print(
            f"  {conv.sample_id}: {len(conv.sessions)} sessions, "
            f"{len(conv.qa)} QA pairs, shape={emb_shape}"
        )

    print(f"\nDone. Saved {saved} new embeddings to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_file", type=str,
                        default="experiments/locomo/data/locomo10.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/locomo/cached_embs")
    parser.add_argument("--max_ctx_tokens", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true")
    precompute(parser.parse_args())
