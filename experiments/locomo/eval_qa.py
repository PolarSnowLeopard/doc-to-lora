"""
LoCoMo QA 评测脚本。

支持多种模式:
  - standalone:  无上下文，基座模型直接回答
  - full_context: 全部对话历史塞进 prompt
  - cmp:         CMP gate 合并所有 session → LoRA → 生成回答
  - d2l_single:  仅用最后一个 session 的 LoRA

评测指标: token-level F1（与 LoCoMo 官方一致）
按 5 类 QA 分别报告: single-hop / multi-hop / temporal / commonsense / adversarial

用法:
  uv run experiments/locomo/eval_qa.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --emb_dir experiments/locomo/cached_embs \
    --mode cmp \
    --cmp_checkpoint experiments/msc/cmp_runs/run1/best_gate.pt
"""
import argparse
import json
import os
import string
import sys
import unicodedata
from collections import Counter
from functools import partial

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../perma"))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers, lora_forward
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.utils import get_layers, get_peft_modules

from cmp import CMPGate

QA_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "commonsense", 4: "single-hop", 5: "adversarial"}


# ---------- F1 evaluation (aligned with LoCoMo official) ----------

def normalize_answer(s):
    s = s.replace(",", "")
    def remove_articles(text):
        import re
        return re.sub(r"\b(a|an|the|and)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score_single(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_qa_f1(prediction, answer, category):
    """与 LoCoMo 官方 eval_question_answering 对齐。"""
    if category == 5:  # adversarial
        pred_lower = prediction.lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0
    if category == 3:  # commonsense: use first part before ';'
        answer = answer.split(";")[0].strip()
    if category == 1:  # multi-hop: split by comma, partial F1
        preds = [p.strip() for p in prediction.split(",")]
        gts = [g.strip() for g in answer.split(",")]
        import numpy as np
        return float(np.mean([max(f1_score_single(p, g) for p in preds) for g in gts]))
    return f1_score_single(prediction, answer)


# ---------- Model helpers ----------

def disable_lora_hooks(model):
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


def apply_lora(model, lora_dict):
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


def gate_to_lora_dict(gate, hypernet, embs, device):
    h = torch.zeros_like(embs[0]).to(device)
    for emb in embs:
        h = gate(h, emb.to(device))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = hypernet.layers(h)
        norm = torch.norm(h, dim=-1, keepdim=True)
        h = h / norm
        flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


def generate_answer(model, tokenizer, question, max_new_tokens=128):
    """用 chat template 格式化 question 并生成回答。"""
    chat = [{"role": "user", "content": f"Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------- Main ----------

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

    print(f"\nLoading embeddings from {args.emb_dir} ...")
    conversations = load_cached_conversations(args.emb_dir)
    print(f"Loaded {len(conversations)} conversations\n")

    all_results = []
    cat_scores = {c: [] for c in QA_CATEGORIES}

    if args.conv_ids:
        conversations = [c for c in conversations if c["sample_id"] in args.conv_ids]
        print(f"Filtered to {len(conversations)} conversations: {[c['sample_id'] for c in conversations]}")

    for conv in conversations:
        n_qa = len(conv["qa"])
        print(f"  {conv['sample_id']}: {conv['n_sessions']} sessions, {n_qa} QA pairs")

        lora_dict = None
        if args.mode == "cmp":
            lora_dict = gate_to_lora_dict(
                gate, model.hypernet, conv["embs"], model.device,
            )
        elif args.mode == "d2l_single":
            emb = conv["embs"][-1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = model.hypernet.layers(emb.to(model.device))
                norm = torch.norm(h, dim=-1, keepdim=True)
                h = h / norm
                flat_loras = model.hypernet.head(h)
            lora_dict = model.hypernet._to_lora_dict(flat_loras)

        for qa in conv["qa"]:
            question = qa["question"]
            gold_answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
            category = qa["category"]

            if args.mode == "full_context":
                ctx = "\n\n".join(
                    f"[Session {i+1}]\n{s}"
                    for i, s in enumerate(conv["session_texts"])
                )
                chat = [{"role": "user", "content": (
                    f"Based on the following conversation history, answer the question concisely.\n\n"
                    f"Conversation:\n{ctx}\n\n"
                    f"Question: {question}\nAnswer:"
                )}]
                input_ids = tokenizer.apply_chat_template(
                    chat, add_special_tokens=False,
                    add_generation_prompt=True, return_tensors="pt",
                ).to(model.device)
                if input_ids.size(1) > args.max_tokens:
                    input_ids = input_ids[:, -args.max_tokens:]
                with torch.no_grad():
                    output_ids = model.base_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                pred = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
                ).strip()

            elif args.mode in ("cmp", "d2l_single"):
                apply_lora(model, lora_dict)
                pred = generate_answer(model, tokenizer, question)
                reset_lora_hooks(model)

            elif args.mode == "standalone":
                pred = generate_answer(model, tokenizer, question)

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            f1 = compute_qa_f1(pred, gold_answer, category)
            cat_scores[category].append(f1)
            all_results.append({
                "sample_id": conv["sample_id"],
                "question": question,
                "gold": gold_answer,
                "prediction": pred,
                "category": category,
                "f1": round(f1, 4),
            })

            torch.cuda.empty_cache()

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")

    header = "| Category     |  F1   | Count |"
    sep =    "|--------------|-------|-------|"
    print(header)
    print(sep)

    total_f1 = []
    for cat_id in sorted(QA_CATEGORIES.keys()):
        scores = cat_scores[cat_id]
        if scores:
            avg = sum(scores) / len(scores)
            total_f1.extend(scores)
            print(f"| {QA_CATEGORIES[cat_id]:<12} | {avg:.3f} | {len(scores):>5} |")
    print(sep)
    if total_f1:
        overall = sum(total_f1) / len(total_f1)
        print(f"| {'Overall':<12} | {overall:.3f} | {len(total_f1):>5} |")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--emb_dir", type=str,
                        default="experiments/locomo/cached_embs")
    parser.add_argument("--mode", type=str, default="cmp",
                        choices=["standalone", "full_context",
                                 "d2l_single", "cmp"])
    parser.add_argument("--cmp_checkpoint", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--conv_ids", type=str, nargs="*", default=None,
                        help="Only evaluate these conversation IDs (e.g. conv-26)")
    evaluate(parser.parse_args())
