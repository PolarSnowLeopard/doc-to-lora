"""
CMP 训练脚本 — MSC 版本。

与 PERMA 版本的关键区别:
  - 训练目标: 语言模型 CE loss（预测 session T 对话），而非 MCQ 准确率
  - 评测指标: Perplexity (PPL)，而非 Accuracy
  - 数据划分: 使用 MSC 自带的 train/validation split

训练流程:
  1. 加载预计算的 session-level lora_emb
  2. 对每个对话的 session T (T=1,2,3):
     - CMP Gate 递归合并 sessions 0~T-1 → h
     - 冻结的 Head → LoRA A/B
     - 将 LoRA 挂到冻结 base model → forward(session T) → CE loss
  3. 仅 Gate 参数更新

用法:
  uv run experiments/msc/train_cmp.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --emb_dir experiments/msc/cached_embs \
    --output_dir experiments/msc/cmp_runs/run1 \
    --epochs 10 --lr 1e-3
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


def load_cached_dialogues(emb_dir: str, split: str):
    """从磁盘加载预计算的 MSC lora_emb 数据。"""
    split_dir = os.path.join(emb_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    dialogues = []
    for dlg_name in sorted(os.listdir(split_dir)):
        dlg_dir = os.path.join(split_dir, dlg_name)
        meta_path = os.path.join(dlg_dir, "meta.pt")
        if not os.path.exists(meta_path):
            continue

        meta = torch.load(meta_path, weights_only=False)
        embs = []
        for s_idx in range(meta["n_sessions"]):
            emb = torch.load(
                os.path.join(dlg_dir, f"session_{s_idx}.pt"),
                weights_only=True, map_location="cpu",
            )
            embs.append(emb)

        dialogues.append({
            "dialogue_id": meta["dialogue_id"],
            "n_sessions": meta["n_sessions"],
            "session_texts": meta["session_texts"],
            "opening_texts": meta.get("opening_texts", []),
            "embs": embs,
        })

    return dialogues


def build_training_samples(dialogues: list[dict]):
    """展开为 (dialogue, target_session_idx) 训练样本。

    对每个对话的 session T (T >= 1), memory = sessions 0..T-1。
    """
    samples = []
    for dlg in dialogues:
        for T in range(1, dlg["n_sessions"]):
            samples.append({
                "dialogue_id": dlg["dialogue_id"],
                "memory_embs": dlg["embs"][:T],
                "target_text": dlg["session_texts"][T],
                "target_session": T,
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
    """CMP gate 递归合并 + LoRA head → lora_dict（带梯度）"""
    h = torch.zeros_like(embs[0]).to(device)
    for emb in embs:
        h = gate(h, emb.to(device))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = hypernet.layers(h)
        norm = torch.norm(h, dim=-1, keepdim=True)
        h = h / norm
        flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


def forward_with_lora_lm(model, lora_dict, input_ids):
    """用给定 lora_dict 做一次 forward，返回 causal LM loss。"""
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

    input_ids = input_ids.to(model.device)
    outputs = model.base_model(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
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
    gate = CMPGate(d_latent=d_latent, init_bias=args.init_bias).to(model.device)
    print(f"CMP Gate params: {sum(p.numel() for p in gate.parameters()):,}")

    tokenizer = get_tokenizer(model.base_model.name_or_path)

    print(f"Loading cached embeddings from {args.emb_dir} ...")
    train_dlgs = load_cached_dialogues(args.emb_dir, "train")
    val_dlgs = load_cached_dialogues(args.emb_dir, "validation")

    train_samples = build_training_samples(train_dlgs)
    val_samples = build_training_samples(val_dlgs)
    print(f"Train: {len(train_dlgs)} dialogues → {len(train_samples)} samples")
    print(f"Val:   {len(val_dlgs)} dialogues → {len(val_samples)} samples")

    if not train_samples:
        raise RuntimeError("No training samples. Run precompute_embs.py first.")

    patch_for_training(model)

    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.lr, weight_decay=args.wd,
    )
    total_steps = args.epochs * len(train_samples)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    best_val_ppl = float("inf")
    log_fh = open(log_path, "w")

    print(f"\nTraining for {args.epochs} epochs, {total_steps} steps")
    print(f"  lr={args.lr}, init_bias={args.init_bias}, wd={args.wd}")
    print(f"  max_target_tokens={args.max_target_tokens}\n")

    global_step = 0
    for epoch in range(args.epochs):
        gate.train()
        random.shuffle(train_samples)
        epoch_loss = 0.0
        n_train = 0
        t0 = time.time()

        for sample in train_samples:
            target_ids = tokenizer.encode(
                sample["target_text"], add_special_tokens=False,
                return_tensors="pt",
            )
            if target_ids.size(1) > args.max_target_tokens:
                target_ids = target_ids[:, :args.max_target_tokens]
            if target_ids.size(1) < 4:
                continue

            lora_dict = gate_forward(
                gate, model.hypernet, sample["memory_embs"], model.device,
            )
            loss = forward_with_lora_lm(model, lora_dict, target_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            n_train += 1
            global_step += 1

        epoch_loss /= max(n_train, 1)
        elapsed = time.time() - t0

        # --- Validation ---
        gate.eval()
        val_loss_sum = 0.0
        val_n = 0
        session_ppl = {}  # session_idx → (loss_sum, count)

        with torch.no_grad():
            for sample in val_samples:
                target_ids = tokenizer.encode(
                    sample["target_text"], add_special_tokens=False,
                    return_tensors="pt",
                )
                if target_ids.size(1) > args.max_target_tokens:
                    target_ids = target_ids[:, :args.max_target_tokens]
                if target_ids.size(1) < 4:
                    continue

                lora_dict = gate_forward(
                    gate, model.hypernet, sample["memory_embs"], model.device,
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

                ids = target_ids.to(model.device)
                logits = model.base_model(input_ids=ids).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                vloss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                reset_lora_hooks(model)
                torch.cuda.empty_cache()

                val_loss_sum += vloss.item()
                val_n += 1

                T = sample["target_session"]
                if T not in session_ppl:
                    session_ppl[T] = [0.0, 0]
                session_ppl[T][0] += vloss.item()
                session_ppl[T][1] += 1

        val_loss = val_loss_sum / max(val_n, 1)
        val_ppl = math.exp(min(val_loss, 20))

        per_session_str = " | ".join(
            f"S{T+1}={math.exp(min(s[0]/max(s[1],1), 20)):.2f}"
            for T, s in sorted(session_ppl.items())
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "per_session_ppl": {
                f"S{T+1}": round(math.exp(min(s[0]/max(s[1],1), 20)), 2)
                for T, s in sorted(session_ppl.items())
            },
            "lr": scheduler.get_last_lr()[0],
            "time": round(elapsed, 1),
        }
        log_fh.write(json.dumps(log_entry) + "\n")
        log_fh.flush()

        marker = ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = os.path.join(args.output_dir, "best_gate.pt")
            torch.save({
                "gate_state_dict": gate.state_dict(),
                "d_latent": d_latent,
                "init_bias": args.init_bias,
                "epoch": epoch + 1,
                "val_ppl": val_ppl,
            }, ckpt_path)
            marker = f" ★ best={val_ppl:.2f}"

        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={epoch_loss:.4f} | "
            f"val_ppl={val_ppl:.2f} ({per_session_str}) | "
            f"{elapsed:.0f}s{marker}"
        )

    log_fh.close()

    final_path = os.path.join(args.output_dir, "final_gate.pt")
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "d_latent": d_latent,
        "init_bias": args.init_bias,
        "epoch": args.epochs,
    }, final_path)
    print(f"\nTraining done. Best val_ppl={best_val_ppl:.2f}")
    print(f"  Best checkpoint: {os.path.join(args.output_dir, 'best_gate.pt')}")
    print(f"  Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="D2L checkpoint")
    parser.add_argument("--emb_dir", type=str,
                        default="experiments/msc/cached_embs")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/msc/cmp_runs/run1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--init_bias", type=float, default=-2.0)
    parser.add_argument("--max_target_tokens", type=int, default=512)
    train(parser.parse_args())
