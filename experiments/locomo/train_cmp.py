"""
CMP 训练脚本 — LoCoMo Leave-One-Out 版本。

训练流程:
  1. 加载预计算的 session-level lora_emb
  2. Leave-one-out: 9 个对话训练, 1 个对话评测
  3. 对每个 QA pair:
     - CMP Gate 递归合并所有 sessions → h
     - 冻结的 Head → LoRA A/B
     - 将 LoRA 挂到冻结 base model → forward(question + gold answer) → CE loss on answer tokens
  4. 仅 Gate 参数更新

用法:
  uv run experiments/locomo/train_cmp.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --emb_dir experiments/locomo/cached_embs \
    --output_dir experiments/locomo/cmp_runs/loo \
    --epochs 20 --lr 1e-3
"""
import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../perma"))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers, lora_forward
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.utils import get_layers, get_peft_modules

from cmp import CMPGate


def load_cached_conversations(emb_dir):
    conversations = []
    for name in sorted(os.listdir(emb_dir)):
        conv_dir = os.path.join(emb_dir, name)
        meta_path = os.path.join(conv_dir, "meta.pt")
        if not os.path.exists(meta_path):
            continue
        meta = torch.load(meta_path, weights_only=False)
        embs = []
        for s_idx in range(meta["n_sessions"]):
            emb = torch.load(
                os.path.join(conv_dir, f"session_{s_idx}.pt"),
                weights_only=True, map_location="cpu",
            )
            embs.append(emb)
        meta["embs"] = embs
        conversations.append(meta)
    return conversations


def build_qa_samples(conversations):
    """展开为 (conversation_embs, question, gold_answer) 训练样本。"""
    samples = []
    for conv in conversations:
        for qa in conv["qa"]:
            answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
            if not answer:
                continue
            samples.append({
                "sample_id": conv["sample_id"],
                "embs": conv["embs"],
                "question": qa["question"],
                "answer": answer,
                "category": qa["category"],
            })
    return samples


def patch_for_training(model):
    layers = get_layers(model.base_model)
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            module = module_info["module"]
            if getattr(module, "patched_forward", False):
                continue
            module.forward_orig = module.forward
            module.patched_forward = True
            module.forward = partial(
                lora_forward, self=module,
                lora_dropout_p=model.peft_config.lora_dropout,
                scaling=model.peft_config.lora_alpha,
            )


def reset_lora_hooks(model):
    layers = get_layers(model.base_model)
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            module = module_info["module"]
            if getattr(module, "patched_forward", False):
                module.forward = partial(
                    lora_forward, self=module,
                    lora_dropout_p=model.peft_config.lora_dropout,
                    scaling=model.peft_config.lora_alpha,
                )


def gate_forward(gate, hypernet, embs, device):
    h = torch.zeros_like(embs[0]).to(device)
    for emb in embs:
        h = gate(h, emb.to(device))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = hypernet.layers(h)
        norm = torch.norm(h, dim=-1, keepdim=True)
        h = h / norm
        flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


def forward_with_lora_qa(model, lora_dict, prompt_ids, answer_ids):
    """Apply LoRA, forward question+answer, compute CE loss on answer tokens only."""
    n_qs = torch.tensor([1], device=model.device)
    lora_combined = combine_lora(
        lora_dict, n_qs,
        lora_bias=model.hypernet.get_head_bias()
        if model.hypernet.config.use_bias else None,
    )
    apply_lora_to_layers(
        model.base_model, model.hypernet.layer_indices,
        lora_combined, n_qs, None,
    )

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
    outputs = model.base_model(input_ids=input_ids)
    logits = outputs.logits

    prompt_len = prompt_ids.size(1)
    answer_logits = logits[:, prompt_len - 1:-1, :].contiguous()
    answer_labels = answer_ids.to(model.device).contiguous()

    loss = F.cross_entropy(
        answer_logits.view(-1, answer_logits.size(-1)),
        answer_labels.view(-1),
    )

    reset_lora_hooks(model)
    return loss


def train(args):
    print(f"Loading D2L model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    for param in model.parameters():
        param.requires_grad = False

    d_latent = model.hypernet.config.latent_size
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    print(f"Loading cached embeddings from {args.emb_dir} ...")
    all_conversations = load_cached_conversations(args.emb_dir)
    conv_ids = [c["sample_id"] for c in all_conversations]
    print(f"Loaded {len(all_conversations)} conversations: {conv_ids}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_fold_results = []

    for fold_idx, held_out_id in enumerate(conv_ids):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(conv_ids)}: held-out = {held_out_id}")
        print(f"{'='*60}")

        train_convs = [c for c in all_conversations if c["sample_id"] != held_out_id]
        val_conv = [c for c in all_conversations if c["sample_id"] == held_out_id][0]

        train_samples = build_qa_samples(train_convs)
        val_qas = val_conv["qa"]

        print(f"  Train: {len(train_convs)} convs → {len(train_samples)} QA samples")
        print(f"  Val: {held_out_id} → {len(val_qas)} QA pairs")

        gate = CMPGate(d_latent=d_latent, init_bias=args.init_bias).to(model.device)
        patch_for_training(model)

        optimizer = torch.optim.AdamW(
            gate.parameters(), lr=args.lr, weight_decay=args.wd,
        )
        total_steps = args.epochs * len(train_samples)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            gate.train()
            random.shuffle(train_samples)
            epoch_loss = 0.0
            n_train = 0
            t0 = time.time()

            for sample in train_samples:
                chat = [{"role": "user", "content": (
                    f"Answer the following question concisely.\n\n"
                    f"Question: {sample['question']}\nAnswer:"
                )}]
                prompt_ids = tokenizer.apply_chat_template(
                    chat, add_special_tokens=False,
                    add_generation_prompt=True, return_tensors="pt",
                )
                answer_ids = tokenizer.encode(
                    " " + sample["answer"], add_special_tokens=False,
                    return_tensors="pt",
                )
                if prompt_ids.size(1) + answer_ids.size(1) > args.max_seq_len:
                    continue

                lora_dict = gate_forward(
                    gate, model.hypernet, sample["embs"], model.device,
                )
                loss = forward_with_lora_qa(model, lora_dict, prompt_ids, answer_ids)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                epoch_loss += loss.item()
                n_train += 1

            epoch_loss /= max(n_train, 1)
            elapsed = time.time() - t0

            # --- Validation: compute loss on held-out QA ---
            gate.eval()
            val_loss_sum = 0.0
            val_n = 0

            with torch.no_grad():
                val_embs = val_conv["embs"]
                lora_dict = gate_forward(
                    gate, model.hypernet, val_embs, model.device,
                )

                for qa in val_qas:
                    answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
                    if not answer:
                        continue

                    chat = [{"role": "user", "content": (
                        f"Answer the following question concisely.\n\n"
                        f"Question: {qa['question']}\nAnswer:"
                    )}]
                    prompt_ids = tokenizer.apply_chat_template(
                        chat, add_special_tokens=False,
                        add_generation_prompt=True, return_tensors="pt",
                    )
                    answer_ids = tokenizer.encode(
                        " " + answer, add_special_tokens=False,
                        return_tensors="pt",
                    )

                    n_qs = torch.tensor([1], device=model.device)
                    lora_combined = combine_lora(
                        lora_dict, n_qs,
                        lora_bias=model.hypernet.get_head_bias()
                        if model.hypernet.config.use_bias else None,
                    )
                    apply_lora_to_layers(
                        model.base_model, model.hypernet.layer_indices,
                        lora_combined, n_qs, None,
                    )

                    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
                    logits = model.base_model(input_ids=input_ids).logits
                    prompt_len = prompt_ids.size(1)
                    a_logits = logits[:, prompt_len - 1:-1, :].contiguous()
                    a_labels = answer_ids.to(model.device).contiguous()
                    vloss = F.cross_entropy(
                        a_logits.view(-1, a_logits.size(-1)),
                        a_labels.view(-1),
                    )

                    reset_lora_hooks(model)
                    val_loss_sum += vloss.item()
                    val_n += 1
                    torch.cuda.empty_cache()

            val_loss = val_loss_sum / max(val_n, 1)
            val_ppl = math.exp(min(val_loss, 20))

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(
                    args.output_dir, f"best_gate_{held_out_id}.pt",
                )
                torch.save({
                    "gate_state_dict": gate.state_dict(),
                    "d_latent": d_latent,
                    "init_bias": args.init_bias,
                    "epoch": epoch + 1,
                    "held_out": held_out_id,
                    "val_loss": val_loss,
                }, ckpt_path)
                marker = f" ★ best"

            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"train_loss={epoch_loss:.4f} | "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | "
                f"{elapsed:.0f}s{marker}"
            )

        all_fold_results.append({
            "held_out": held_out_id,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_ppl": round(math.exp(min(best_val_loss, 20)), 2),
            "n_train_samples": len(train_samples),
            "n_val_qa": len(val_qas),
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Leave-One-Out Summary")
    print(f"{'='*60}")
    for r in all_fold_results:
        print(f"  {r['held_out']}: val_ppl={r['best_val_ppl']:.2f} "
              f"(train={r['n_train_samples']}, val={r['n_val_qa']})")

    avg_ppl = math.exp(
        sum(r["best_val_loss"] for r in all_fold_results) / len(all_fold_results)
    )
    print(f"\n  Average val_ppl: {avg_ppl:.2f}")

    summary_path = os.path.join(args.output_dir, "loo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--emb_dir", type=str,
                        default="experiments/locomo/cached_embs")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/locomo/cmp_runs/loo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--init_bias", type=float, default=-2.0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    train(parser.parse_args())
