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
import traceback

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

MAX_CTX_TOKENS = 7000


def build_mcq_prompt(question: str, options: list[str]) -> str:
    n = len(options)
    max_letter = chr(65 + n - 1)
    opts_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Answer with the letter only (A-{max_letter}):"
    )


def extract_answer(text: str, n_options: int) -> str:
    max_letter = chr(65 + n_options - 1)
    text = text.strip()
    match = re.search(rf"\b([A-{max_letter}])\b", text)
    return match.group(1) if match else ""


def safe_internalize(model, text, tokenizer):
    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    tokens = ctx_tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > MAX_CTX_TOKENS:
        text = ctx_tokenizer.decode(tokens[:MAX_CTX_TOKENS], skip_special_tokens=True)
    model.internalize(text)


def ask_mcq(model, tokenizer, task: PermaTask) -> str:
    prompt = build_mcq_prompt(task.question, task.options)
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, max_new_tokens=16)
    new_tokens = out[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"    [DEBUG] new_tokens={new_tokens.tolist()} raw='{raw_text}' clean='{generated_text}'")
    return extract_answer(generated_text, len(task.options))


def evaluate_task_oracle(
    model, tokenizer, task: PermaTask,
) -> dict:
    """Oracle: 拼全部 session 一次性 internalize"""
    full_text = sessions_to_full_text(task.sessions)
    model.reset()
    safe_internalize(model, full_text, tokenizer)
    pred = ask_mcq(model, tokenizer, task)

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
    safe_internalize(model, last_text, tokenizer)
    pred = ask_mcq(model, tokenizer, task)

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
        safe_internalize(model, text, tokenizer)
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
    pred = ask_mcq(model, tokenizer, task)

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
    }


def evaluate_task_no_lora(
    model, tokenizer, task: PermaTask,
) -> dict:
    """No LoRA: 基座模型 + 截断上下文，验证模型能否正常回答 MCQ"""
    model.reset()
    prompt = build_mcq_prompt(task.question, task.options)
    last_session = session_to_text(task.sessions[-1])
    # 截断上下文确保总长度在 7K token 以内（留 1K 给 prompt + 生成）
    ctx_tokens = tokenizer.encode(last_session, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    max_ctx = 7000 - len(prompt_tokens) - 100
    if len(ctx_tokens) > max_ctx:
        last_session = tokenizer.decode(ctx_tokens[:max(max_ctx, 500)], skip_special_tokens=True)

    content = f"Based on the following conversation, answer the question.\n\nConversation:\n{last_session}\n\n{prompt}"
    chat = [{"role": "user", "content": content}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    print(f"    [DEBUG no_lora] input_len={input_ids.shape[-1]}")

    with torch.inference_mode():
        out = model.base_model.generate(input_ids=input_ids, max_new_tokens=16)
    new_tokens = out[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"    [DEBUG no_lora] raw='{raw_text}' clean='{generated_text}'")
    pred = extract_answer(generated_text, len(task.options))

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
        "no_lora": evaluate_task_no_lora,
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Running: {mode}")
        print(f"{'='*60}")

        results = []
        for i, task in enumerate(tasks):
            try:
                res = eval_fn[mode](model, tokenizer, task)
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] ERROR task={task.task_id}: {e}")
                traceback.print_exc()
                res = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "pred": "",
                    "gold": task.gold_label,
                    "correct": False,
                    "error": str(e),
                }
            results.append(res)
            status = "✓" if res["correct"] else "✗"
            print(f"  [{i+1}/{len(tasks)}] {status} task={task.task_id} type={task.task_type} pred={res['pred']} gold={res['gold']}")

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
                        choices=["all", "oracle", "single_shot", "naive_merge", "no_lora"])
    parser.add_argument("--max_users", type=int, default=2,
                        help="Max users to evaluate (0 = all)")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/perma/results")
    run_diagnostic(parser.parse_args())
