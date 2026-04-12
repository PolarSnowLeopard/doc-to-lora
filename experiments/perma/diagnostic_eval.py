"""
诊断实验：在 PERMA 上对比 Doc-to-LoRA 的三种使用方式，
验证"历史记忆是否重要"以及"naive merge 是否有效"。

三组对比：
  (a) Oracle:      拼全部 session → internalize（上界）
  (b) Single-shot: 只用最新 session → internalize（下界）
  (c) Naive merge: 每个 session 独立 internalize → LoRA 参数平均

使用方式：
  PERMA_DATA_ROOT=/path/to/perma/data \
  python experiments/perma/diagnostic_eval.py \
    --checkpoint trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
    --mode all \
    --max_users 2
"""
import argparse
import copy
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from data_adapter import (
    PermaTask,
    load_tasks,
    session_to_text,
    sessions_to_full_text,
    ALL_USER_IDS,
)


def build_mcq_prompt(question: str, options: list[str]) -> str:
    opts_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Answer with the letter only (A, B, C, or D):"
    )


def extract_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else ""


def evaluate_task_oracle(
    model, tokenizer, task: PermaTask,
) -> dict:
    """Oracle: 拼全部 session 一次性 internalize"""
    full_text = sessions_to_full_text(task.sessions)
    model.reset()
    model.internalize(full_text)

    prompt = build_mcq_prompt(task.question, task.options)
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, max_new_tokens=16)
    pred = extract_answer(tokenizer.decode(out[0], skip_special_tokens=True))

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
        "num_sessions": len(task.sessions),
    }


def evaluate_task_single_shot(
    model, tokenizer, task: PermaTask,
) -> dict:
    """Single-shot: 只用最后一个 session"""
    last_text = session_to_text(task.sessions[-1])
    model.reset()
    model.internalize(last_text)

    prompt = build_mcq_prompt(task.question, task.options)
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, max_new_tokens=16)
    pred = extract_answer(tokenizer.decode(out[0], skip_special_tokens=True))

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
    }


def evaluate_task_naive_merge(
    model, tokenizer, task: PermaTask,
) -> dict:
    """Naive merge: 每个 session 独立 internalize，LoRA 参数取平均"""
    all_loras = []
    for session in task.sessions:
        text = session_to_text(session)
        model.reset()
        model.internalize(text)
        lora_copy = copy.deepcopy(model.generated_loras)
        all_loras.append(lora_copy)

    if len(all_loras) > 1:
        merged = {}
        for key in all_loras[0]:
            merged[key] = {}
            for mat in all_loras[0][key]:  # 'A', 'B'
                stacked = torch.stack([l[key][mat] for l in all_loras])
                merged[key][mat] = stacked.mean(dim=0)
        model.generated_loras = merged
    else:
        model.generated_loras = all_loras[0]

    model.patch_lora_forward()
    prompt = build_mcq_prompt(task.question, task.options)
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, max_new_tokens=16)
    pred = extract_answer(tokenizer.decode(out[0], skip_special_tokens=True))

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
    }


def run_diagnostic(args):
    print(f"Loading model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
        use_flash_attn=False,
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    user_ids = ALL_USER_IDS[:args.max_users] if args.max_users > 0 else None
    tasks = load_tasks(user_ids=user_ids, noise=False, multi_domain=False)
    print(f"Loaded {len(tasks)} tasks from {len(set(t.user_id for t in tasks))} users")

    modes = (
        ["oracle", "single_shot", "naive_merge"]
        if args.mode == "all"
        else [args.mode]
    )

    eval_fn = {
        "oracle": evaluate_task_oracle,
        "single_shot": evaluate_task_single_shot,
        "naive_merge": evaluate_task_naive_merge,
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Running: {mode}")
        print(f"{'='*60}")

        results = []
        for i, task in enumerate(tasks):
            res = eval_fn[mode](model, tokenizer, task)
            results.append(res)
            status = "✓" if res["correct"] else "✗"
            print(f"  [{i+1}/{len(tasks)}] {status} task={task.task_id} type={task.task_type} pred={res['pred']} gold={res['gold']}")

        # 按 task_type 分组统计
        acc_all = sum(r["correct"] for r in results) / max(len(results), 1)
        print(f"\n  Overall accuracy: {acc_all:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")

        for t in sorted(set(r["task_type"] for r in results)):
            group = [r for r in results if r["task_type"] == t]
            acc = sum(r["correct"] for r in group) / max(len(group), 1)
            type_names = {1: "Zero-Memory", 2: "In-Time", 3: "Post-Intervention"}
            print(f"  Type {t} ({type_names.get(t, '?')}): {acc:.3f} ({sum(r['correct'] for r in group)}/{len(group)})")

        out_path = os.path.join(args.output_dir, f"diagnostic_{mode}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "oracle", "single_shot", "naive_merge"])
    parser.add_argument("--max_users", type=int, default=2,
                        help="Max users to evaluate (0 = all)")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/perma/results")
    run_diagnostic(parser.parse_args())
