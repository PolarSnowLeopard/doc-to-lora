"""
预计算 MSC 各 session 的 D2L Aggregator 输出（lora_emb）。

目录结构: {output_dir}/{split}/dialogue_{id}/
  - session_0.pt, session_1.pt, ...
  - meta.pt  (dialogue_id, split, n_sessions, session_texts, ...)

用法:
  uv run experiments/msc/precompute_embs.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --output_dir experiments/msc/cached_embs \
    --splits train validation test
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from data_adapter import load_dialogues


def precompute(args):
    print(f"Loading D2L model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.eval()

    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        dialogues = load_dialogues(
            split=split, cache_dir=args.cache_dir,
            max_dialogues=args.max_dialogues,
        )
        print(f"  Loaded {len(dialogues)} dialogues")

        saved = 0
        for i, dlg in enumerate(dialogues):
            dlg_dir = os.path.join(
                args.output_dir, split, f"dialogue_{dlg.dialogue_id}",
            )
            os.makedirs(dlg_dir, exist_ok=True)

            session_texts = []
            emb_shape = None

            for s_idx, session in enumerate(dlg.sessions):
                cache_path = os.path.join(dlg_dir, f"session_{s_idx}.pt")
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

            opening_texts = []
            for session in dlg.sessions:
                if session["utterances"]:
                    opening_texts.append(
                        f"{session['speakers'][0]}: {session['utterances'][0]}"
                    )
                else:
                    opening_texts.append("")

            meta = {
                "dialogue_id": dlg.dialogue_id,
                "split": split,
                "n_sessions": len(dlg.sessions),
                "session_texts": session_texts,
                "opening_texts": opening_texts,
                "emb_shape": list(emb_shape) if emb_shape else [],
            }
            torch.save(meta, os.path.join(dlg_dir, "meta.pt"))

            if (i + 1) % 200 == 0 or i == len(dialogues) - 1:
                print(
                    f"  [{i+1}/{len(dialogues)}] dialogue_{dlg.dialogue_id}: "
                    f"{len(dlg.sessions)} sessions, shape={emb_shape}"
                )

        print(f"  Done. Saved {saved} new embeddings for {split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="experiments/msc/cached_embs")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "validation", "test"])
    parser.add_argument("--max_ctx_tokens", type=int, default=4000)
    parser.add_argument("--max_dialogues", type=int, default=0,
                        help="per-split 对话上限 (0=不限)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace datasets 缓存目录")
    parser.add_argument("--overwrite", action="store_true")
    precompute(parser.parse_args())
