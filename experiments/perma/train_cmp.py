"""
CMP (Continual Memory Parametrization) 训练脚本。

训练流程:
  1. 加载预计算的 session-level lora_emb
  2. CMP Gate 递归合并 → h_T
  3. 冻结的 ResMLPBlock + L2Norm + EinMix Head → LoRA A/B
  4. 将 LoRA 挂到冻结的 base model → forward → CE loss（gold MCQ letter）
  5. 仅 Gate 参数更新

用法:
  PERMA_DATA_ROOT=experiments/perma/data \
  python experiments/perma/train_cmp.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --emb_dir experiments/perma/cached_embs \
    --output_dir experiments/perma/cmp_runs \
    --epochs 30 --lr 1e-3
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

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers, lora_forward
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.utils import get_layers, get_peft_modules

from cmp import CMPGate
from data_adapter import ALL_USER_IDS

HELD_OUT_USER = 334


def load_cached_tasks(emb_dir: str, user_ids: list[int] | None = None):
    """从磁盘加载预计算的 lora_emb 数据集。"""
    tasks = []
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"emb_dir not found: {emb_dir}")

    for user_dir in sorted(os.listdir(emb_dir)):
        if not user_dir.startswith("user"):
            continue
        uid = int(user_dir.replace("user", ""))
        if user_ids is not None and uid not in user_ids:
            continue

        user_path = os.path.join(emb_dir, user_dir)
        for task_id in sorted(os.listdir(user_path)):
            task_dir = os.path.join(user_path, task_id)
            meta_path = os.path.join(task_dir, "meta.pt")
            if not os.path.exists(meta_path):
                continue

            meta = torch.load(meta_path, weights_only=False)
            embs = []
            for s_idx in range(meta["n_sessions"]):
                emb = torch.load(
                    os.path.join(task_dir, f"session_{s_idx}.pt"),
                    weights_only=True, map_location="cpu",
                )
                embs.append(emb)

            tasks.append({
                "user_id": uid,
                "task_id": task_id,
                "task_type": meta["task_type"],
                "question": meta["question"],
                "options": meta["options"],
                "gold_label": meta["gold_label"],
                "embs": embs,
            })

    return tasks


def build_prompt_ids(tokenizer, question: str, options: list[str]):
    """构建 MCQ prompt token ids, 返回 (input_ids, gold_token_ids_per_option)"""
    n = len(options)
    max_letter = chr(65 + n - 1)
    opts_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Answer with the letter only (A-{max_letter}):"
    )
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    )
    return input_ids


def get_label_token_ids(tokenizer, n_options: int = 8) -> dict[str, int]:
    """预先映射 A-H 到 token id"""
    mapping = {}
    for i in range(n_options):
        letter = chr(65 + i)
        ids = tokenizer.encode(letter, add_special_tokens=False)
        mapping[letter] = ids[-1]
    return mapping


def patch_for_training(model):
    """确保 lora_forward hook 已安装（但不在 no_grad 下）"""
    layers = get_layers(model.base_model)
    lora_forward_fn = lora_forward
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            module = module_info["module"]
            if getattr(module, "patched_forward", False):
                continue
            module.forward_orig = module.forward
            module.patched_forward = True
            module.forward = partial(
                lora_forward_fn,
                self=module,
                lora_dropout_p=model.peft_config.lora_dropout,
                scaling=model.peft_config.lora_alpha,
            )


def reset_lora_hooks(model):
    """移除临时 LoRA partial forward（恢复到 patched 的基础 lora_forward）"""
    layers = get_layers(model.base_model)
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            module = module_info["module"]
            if getattr(module, "patched_forward", False):
                module.forward = partial(
                    lora_forward,
                    self=module,
                    lora_dropout_p=model.peft_config.lora_dropout,
                    scaling=model.peft_config.lora_alpha,
                )


def forward_with_lora(model, lora_dict, input_ids, label_token_id):
    """用给定 lora_dict 做一次 forward，返回 loss。

    label_token_id: 目标 token (A/B/C/.../H 的 id)
    """
    n_qs = torch.tensor([1], device=model.device)
    lora_combined = combine_lora(
        lora_dict, n_qs,
        lora_bias=model.hypernet.get_head_bias()
        if model.hypernet.config.use_bias else None,
    )
    apply_lora_to_layers(
        model.base_model,
        model.hypernet.layer_indices,
        lora_combined,
        n_qs,
        None,
    )

    input_ids = input_ids.to(model.device)
    outputs = model.base_model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]  # [1, vocab_size]

    target = torch.tensor([label_token_id], device=model.device)
    loss = F.cross_entropy(logits, target)

    reset_lora_hooks(model)
    return loss


def gate_forward(gate, hypernet, embs, device):
    """CMP gate 递归合并 + LoRA head → lora_dict（带梯度）"""
    h = torch.zeros_like(embs[0]).to(device)
    for emb in embs:
        h = gate(h, emb.to(device))

    # 冻结的 ResMLPBlock + L2Norm + EinMix Head
    h = hypernet.layers(h)
    norm = torch.norm(h, dim=-1, keepdim=True)
    h = h / norm
    flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


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
    label_map = get_label_token_ids(tokenizer)

    train_users = [u for u in ALL_USER_IDS if u != HELD_OUT_USER]
    val_users = [HELD_OUT_USER]

    print(f"Loading cached embeddings from {args.emb_dir} ...")
    train_tasks = load_cached_tasks(args.emb_dir, train_users)
    val_tasks = load_cached_tasks(args.emb_dir, val_users)
    print(f"Train: {len(train_tasks)} tasks, Val: {len(val_tasks)} tasks")

    if not train_tasks:
        raise RuntimeError("No training tasks found. Run precompute_embs.py first.")

    patch_for_training(model)

    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train_tasks)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    best_val_acc = 0.0
    log_fh = open(log_path, "w")

    print(f"\nTraining for {args.epochs} epochs, {total_steps} steps")
    print(f"  lr={args.lr}, init_bias={args.init_bias}, wd={args.wd}\n")

    global_step = 0
    for epoch in range(args.epochs):
        gate.train()
        random.shuffle(train_tasks)
        epoch_loss = 0.0
        correct = 0
        t0 = time.time()

        for task in train_tasks:
            gold_tid = label_map.get(task["gold_label"])
            if gold_tid is None:
                continue

            lora_dict = gate_forward(
                gate, model.hypernet, task["embs"], model.device,
            )
            prompt_ids = build_prompt_ids(tokenizer, task["question"], task["options"])
            loss = forward_with_lora(model, lora_dict, prompt_ids, gold_tid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            with torch.no_grad():
                pred_tid = model.base_model(
                    input_ids=prompt_ids.to(model.device)
                ).logits[:, -1, :].argmax(-1).item()
            # 简化：用 loss 跟踪，pred 在 validation 时全量评估
            global_step += 1

        epoch_loss /= len(train_tasks)
        elapsed = time.time() - t0

        # --- Validation ---
        gate.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        type_correct = {1: 0, 2: 0, 3: 0}
        type_total = {1: 0, 2: 0, 3: 0}

        with torch.no_grad():
            for task in val_tasks:
                gold_tid = label_map.get(task["gold_label"])
                if gold_tid is None:
                    continue

                lora_dict = gate_forward(
                    gate, model.hypernet, task["embs"], model.device,
                )
                prompt_ids = build_prompt_ids(
                    tokenizer, task["question"], task["options"],
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

                ids = prompt_ids.to(model.device)
                logits = model.base_model(input_ids=ids).logits[:, -1, :]
                pred_id = logits.argmax(-1).item()
                is_correct = (pred_id == gold_tid)

                vloss = F.cross_entropy(
                    logits, torch.tensor([gold_tid], device=model.device),
                )
                val_loss += vloss.item()
                val_correct += int(is_correct)
                val_total += 1
                tt = task["task_type"]
                type_correct[tt] = type_correct.get(tt, 0) + int(is_correct)
                type_total[tt] = type_total.get(tt, 0) + 1

                reset_lora_hooks(model)
                torch.cuda.empty_cache()

        val_acc = val_correct / max(val_total, 1)
        val_loss /= max(val_total, 1)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_correct": val_correct,
            "val_total": val_total,
            "type1_acc": round(type_correct[1] / max(type_total[1], 1), 3),
            "type2_acc": round(type_correct[2] / max(type_total[2], 1), 3),
            "type3_acc": round(type_correct[3] / max(type_total[3], 1), 3),
            "lr": scheduler.get_last_lr()[0],
            "time": round(elapsed, 1),
        }
        log_fh.write(json.dumps(log_entry) + "\n")
        log_fh.flush()

        type_str = (f"T1={log_entry['type1_acc']:.3f} "
                    f"T2={log_entry['type2_acc']:.3f} "
                    f"T3={log_entry['type3_acc']:.3f}")
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={epoch_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({val_correct}/{val_total}) | "
              f"{type_str} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_gate.pt")
            torch.save({
                "gate_state_dict": gate.state_dict(),
                "d_latent": d_latent,
                "init_bias": args.init_bias,
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"    ★ New best val_acc={val_acc:.3f}, saved to {ckpt_path}")

    log_fh.close()

    final_path = os.path.join(args.output_dir, "final_gate.pt")
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "d_latent": d_latent,
        "init_bias": args.init_bias,
        "epoch": args.epochs,
    }, final_path)
    print(f"\nTraining done. Best val_acc={best_val_acc:.3f}")
    print(f"  Best checkpoint: {os.path.join(args.output_dir, 'best_gate.pt')}")
    print(f"  Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="D2L fine-tuned checkpoint")
    parser.add_argument("--emb_dir", type=str, default="experiments/perma/cached_embs")
    parser.add_argument("--output_dir", type=str, default="experiments/perma/cmp_runs/run1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--init_bias", type=float, default=-2.0)
    train(parser.parse_args())
