"""
MSC Perplexity 评测脚本。

支持多种评测模式:
  - standalone:   无上下文，直接 forward session T 文本
  - full_context: sessions 0~T-1 全文拼接到 session T 前面
  - d2l_single:   仅使用 session T-1 的 LoRA (D2L single-shot)
  - cmp:          CMP gate 合并 sessions 0~T-1 的 LoRA

输出格式与 MSC 原始论文 Table 7 对齐:
  按 session 报告 PPL (Session 2/3/4) + Session Openings PPL

用法:
  uv run experiments/msc/eval_ppl.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --emb_dir experiments/msc/cached_embs \
    --mode cmp \
    --cmp_checkpoint experiments/msc/cmp_runs/run1/best_gate.pt
"""
import argparse
import math
import os
import sys
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


def disable_lora_hooks(model):
    """移除 D2L 加载时注入的 lora_forward hooks，恢复原始 nn.Linear.forward。"""
    layers = get_layers(model.base_model)
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            module = module_info["module"]
            module.forward = torch.nn.Linear.forward.__get__(module, type(module))


def patch_model(model):
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


def compute_ppl(model, input_ids):
    """Compute PPL on a token sequence (no LoRA applied here)."""
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        logits = model.base_model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss.item()


def compute_ppl_with_lora(model, lora_dict, input_ids):
    """Compute PPL on a token sequence with LoRA applied."""
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
    with torch.no_grad():
        logits = model.base_model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    reset_lora_hooks(model)
    return loss.item()


def gate_to_lora_dict(gate, hypernet, embs, device):
    """CMP gate → lora_dict (frozen, no grad)."""
    h = torch.zeros_like(embs[0]).to(device)
    for emb in embs:
        h = gate(h, emb.to(device))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = hypernet.layers(h)
        norm = torch.norm(h, dim=-1, keepdim=True)
        h = h / norm
        flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


def evaluate(args):
    print(f"Loading D2L model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.eval()

    tokenizer = get_tokenizer(model.base_model.name_or_path)

    gate = None
    if args.mode == "cmp":
        if not args.cmp_checkpoint:
            raise ValueError("--cmp_checkpoint required for mode=cmp")
        ckpt = torch.load(args.cmp_checkpoint, weights_only=False)
        gate = CMPGate(
            d_latent=ckpt["d_latent"],
            init_bias=ckpt.get("init_bias", -2.0),
        ).to(model.device)
        gate.load_state_dict(ckpt["gate_state_dict"])
        gate.eval()
        print(f"Loaded CMP gate from {args.cmp_checkpoint}")

    if args.mode in ("cmp", "d2l_single"):
        patch_model(model)
    else:
        disable_lora_hooks(model)

    print(f"\nLoading test embeddings from {args.emb_dir} ...")
    dialogues = load_cached_dialogues(args.emb_dir, args.split)
    print(f"Loaded {len(dialogues)} dialogues\n")

    # session_idx → list of (loss, n_tokens)
    session_losses = {}
    opening_losses = {}
    total_loss = 0.0
    total_n = 0

    for i, dlg in enumerate(dialogues):
        for T in range(1, dlg["n_sessions"]):
            target_text = dlg["session_texts"][T]
            target_ids = tokenizer.encode(
                target_text, add_special_tokens=False, return_tensors="pt",
            )
            if target_ids.size(1) < 4:
                continue

            # --- compute loss based on mode ---
            if args.mode == "standalone":
                loss = compute_ppl(model, target_ids)

            elif args.mode == "full_context":
                ctx_text = "\n\n".join(
                    f"[Session {s+1}]\n{dlg['session_texts'][s]}"
                    for s in range(T)
                )
                full_text = ctx_text + "\n\n" + f"[Session {T+1}]\n" + target_text
                full_ids = tokenizer.encode(
                    full_text, add_special_tokens=False, return_tensors="pt",
                )
                if full_ids.size(1) > args.max_tokens:
                    full_ids = full_ids[:, -args.max_tokens:]

                full_ids = full_ids.to(model.device)
                with torch.no_grad():
                    logits = model.base_model(input_ids=full_ids).logits

                ctx_ids = tokenizer.encode(
                    ctx_text + "\n\n" + f"[Session {T+1}]\n",
                    add_special_tokens=False,
                )
                ctx_len = min(len(ctx_ids), full_ids.size(1) - 1)
                shift_logits = logits[:, ctx_len:-1, :].contiguous()
                shift_labels = full_ids[:, ctx_len+1:].contiguous()
                if shift_labels.numel() == 0:
                    continue
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).item()

            elif args.mode == "d2l_single":
                emb = dlg["embs"][T - 1]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h = model.hypernet.layers(emb.to(model.device))
                    norm = torch.norm(h, dim=-1, keepdim=True)
                    h = h / norm
                    flat_loras = model.hypernet.head(h)
                lora_dict = model.hypernet._to_lora_dict(flat_loras)
                loss = compute_ppl_with_lora(model, lora_dict, target_ids)

            elif args.mode == "cmp":
                memory_embs = dlg["embs"][:T]
                lora_dict = gate_to_lora_dict(
                    gate, model.hypernet, memory_embs, model.device,
                )
                loss = compute_ppl_with_lora(model, lora_dict, target_ids)

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            if T not in session_losses:
                session_losses[T] = []
            session_losses[T].append(loss)
            total_loss += loss
            total_n += 1

            # --- session opening PPL ---
            opening_text = dlg["opening_texts"][T] if T < len(dlg.get("opening_texts", [])) else ""
            if opening_text:
                open_ids = tokenizer.encode(
                    opening_text, add_special_tokens=False, return_tensors="pt",
                )
                if open_ids.size(1) >= 4:
                    if args.mode == "standalone":
                        o_loss = compute_ppl(model, open_ids)
                    elif args.mode == "full_context":
                        o_loss = loss  # approximate
                    elif args.mode in ("d2l_single", "cmp"):
                        o_loss = compute_ppl_with_lora(
                            model,
                            lora_dict if args.mode == "cmp" else
                            model.hypernet._to_lora_dict(flat_loras),
                            open_ids,
                        )
                    else:
                        o_loss = loss

                    if T not in opening_losses:
                        opening_losses[T] = []
                    opening_losses[T].append(o_loss)

            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == len(dialogues) - 1:
            avg = math.exp(min(total_loss / max(total_n, 1), 20))
            print(f"  [{i+1}/{len(dialogues)}] running avg PPL={avg:.2f}")

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode} | Split: {args.split}")
    print(f"{'='*60}")

    header = "| Session |  PPL  | Opening PPL | Count |"
    sep = "|---------|-------|-------------|-------|"
    print(header)
    print(sep)

    for T in sorted(session_losses.keys()):
        s_losses = session_losses[T]
        avg_loss = sum(s_losses) / len(s_losses)
        ppl = math.exp(min(avg_loss, 20))

        o_losses = opening_losses.get(T, [])
        if o_losses:
            o_avg = sum(o_losses) / len(o_losses)
            o_ppl = math.exp(min(o_avg, 20))
            o_str = f"{o_ppl:>10.2f}"
        else:
            o_str = "       N/A"

        print(f"|    S{T+1}   | {ppl:>5.2f} | {o_str}  | {len(s_losses):>5d} |")

    overall_loss = total_loss / max(total_n, 1)
    overall_ppl = math.exp(min(overall_loss, 20))
    all_openings = [l for ls in opening_losses.values() for l in ls]
    if all_openings:
        overall_o = math.exp(min(sum(all_openings)/len(all_openings), 20))
        o_all_str = f"{overall_o:>10.2f}"
    else:
        o_all_str = "       N/A"
    print(sep)
    print(f"|   All   | {overall_ppl:>5.2f} | {o_all_str}  | {total_n:>5d} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--emb_dir", type=str,
                        default="experiments/msc/cached_embs")
    parser.add_argument("--mode", type=str, default="cmp",
                        choices=["standalone", "full_context",
                                 "d2l_single", "cmp"])
    parser.add_argument("--cmp_checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_tokens", type=int, default=4096)
    evaluate(parser.parse_args())
