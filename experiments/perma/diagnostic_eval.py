"""
PERMA 评测脚本：对比多种记忆方法在 PERMA benchmark 上的表现。

评测模式：
  Doc-to-LoRA 系列:
    oracle       — 全部 session 拼接 → internalize → LoRA
    single_shot  — 只用最新 session → internalize → LoRA
    naive_merge  — 每个 session 独立 internalize → LoRA 参数平均
  基线方法:
    standalone   — 全部对话拼进 prompt（PERMA 原版格式），基座模型直接回答
    rag          — BGE-M3 检索 top-k 对话片段作为上下文，基座模型回答
    no_lora      — 仅最后 session 拼进 prompt（简化 standalone）

使用方式：
  PERMA_DATA_ROOT=/path/to/perma/data \
  python experiments/perma/diagnostic_eval.py \
    --checkpoint trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
    --mode standalone \
    --max_users 1
"""
import argparse
import copy
import json
import os
import re
import sys
import traceback

import numpy as np
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

MAX_CTX_TOKENS = 4000

# --------------- PERMA 原版 MCQ Prompt ---------------
PERMA_ANSWER_PROMPT = """You are an assistant specialized in answering multiple-choice questions.

## Your Memory
{context}

## User Task Query
{question}

## Options:
{options}

Your goal is to choose **the most appropriate answer option for the User Task Query** from the Options based on your memory. The output should be **ONLY the option key** without any additional explanation, e.g. `A`, etc.

Your response:
"""


def format_options_text(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))


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
        lora_copy = {
            k: {m: v.detach().clone() for m, v in mats.items()}
            for k, mats in model.generated_loras.items()
        }
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


def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def evaluate_task_no_lora(
    model, tokenizer, task: PermaTask,
) -> dict:
    """No LoRA: 基座模型 + 上下文放 prompt，验证模型能否正常回答 MCQ"""
    model.reset()
    prompt = build_mcq_prompt(task.question, task.options)
    last_session = session_to_text(task.sessions[-1])
    tmpl = f"Based on the following conversation, answer the question.\n\nConversation:\n{{CTX}}\n\n{prompt}"
    tmpl_tokens = len(tokenizer.encode(tmpl, add_special_tokens=False))
    max_ctx = max(7500 - tmpl_tokens, 500)
    last_session = truncate_text_to_tokens(last_session, tokenizer, max_ctx)

    content = tmpl.replace("{CTX}", last_session)
    chat = [{"role": "user", "content": content}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    print(f"    [DEBUG no_lora] input_len={input_ids.shape[-1]}")

    with torch.inference_mode():
        out = model.base_model.generate(
            input_ids=input_ids, max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
        )
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


# --------------- Standalone baseline ---------------

def flatten_all_messages(task: PermaTask) -> str:
    """将所有 session 的原始对话拼为 PERMA standalone 格式的上下文"""
    lines = []
    for conv in task.raw_conversations:
        for msg in conv:
            lines.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(lines)


def evaluate_task_standalone(
    model, tokenizer, task: PermaTask,
) -> dict:
    """Standalone: 全部对话放进 prompt，基座模型直接回答（PERMA 原版格式）"""
    model.reset()
    context = flatten_all_messages(task)
    options_text = format_options_text(task.options)

    prompt_text = PERMA_ANSWER_PROMPT.format(
        context=context,
        question=task.question,
        options=options_text,
    )
    chat = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    # 截断到模型最大长度（保留尾部，即问题+选项部分）
    max_len = getattr(tokenizer, "model_max_length", 32768)
    if max_len > 100000:
        max_len = 32768
    if input_ids.shape[-1] > max_len - 32:
        input_ids = input_ids[:, -(max_len - 32):]

    print(f"    [DEBUG standalone] input_len={input_ids.shape[-1]}")

    with torch.inference_mode():
        out = model.base_model.generate(
            input_ids=input_ids, max_new_tokens=16,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"    [DEBUG standalone] raw='{raw_text}' clean='{generated_text}'")
    pred = extract_answer(generated_text, len(task.options))

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
    }


# --------------- RAG baseline ---------------

_rag_model = None


def get_rag_model():
    global _rag_model
    if _rag_model is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _rag_model = SentenceTransformer("BAAI/bge-m3", device=device)
    return _rag_model


def evaluate_task_rag(
    model, tokenizer, task: PermaTask,
    top_k: int = 10, batch_size: int = 2,
) -> dict:
    """RAG: BGE-M3 编码对话片段 → top-k 检索 → 基座模型回答"""
    model.reset()
    emb_model = get_rag_model()

    all_messages = []
    for conv in task.raw_conversations:
        all_messages.extend(conv)

    chunks = []
    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i + batch_size]
        chunk_text = "\n".join(f"{m['role']}: {m['content']}" for m in batch)
        chunks.append(chunk_text)

    if not chunks:
        context = ""
    else:
        chunk_embeddings = emb_model.encode(chunks, normalize_embeddings=True)
        q_vec = emb_model.encode([task.question], normalize_embeddings=True)[0]
        sims = np.dot(chunk_embeddings, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        context = "\n\n".join(chunks[i] for i in sorted(top_indices))

    options_text = format_options_text(task.options)
    prompt_text = PERMA_ANSWER_PROMPT.format(
        context=context,
        question=task.question,
        options=options_text,
    )
    chat = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_special_tokens=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    max_len = getattr(tokenizer, "model_max_length", 32768)
    if max_len > 100000:
        max_len = 32768
    if input_ids.shape[-1] > max_len - 32:
        input_ids = input_ids[:, -(max_len - 32):]

    print(f"    [DEBUG rag] input_len={input_ids.shape[-1]} chunks={len(chunks)} top_k={min(top_k, len(chunks))}")

    with torch.inference_mode():
        out = model.base_model.generate(
            input_ids=input_ids, max_new_tokens=16,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"    [DEBUG rag] raw='{raw_text}' clean='{generated_text}'")
    pred = extract_answer(generated_text, len(task.options))

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
    }


def evaluate_task_cmp(
    model, tokenizer, task: PermaTask,
    gate=None,
) -> dict:
    """CMP: 逐 session 过 Encoder+Aggregator → Gate 合并 → Head → LoRA → 回答"""
    from cmp import CMPGate, extract_aggregator_output, lora_emb_to_lora_dict, run_cmp_sessions

    session_embs = []
    for session in task.sessions:
        text = session_to_text(session)
        emb = extract_aggregator_output(model, text, max_tokens=MAX_CTX_TOKENS)
        session_embs.append(emb)

    h = run_cmp_sessions(gate, session_embs)
    lora_dict = lora_emb_to_lora_dict(model.hypernet, h)

    model.reset()
    model.generated_loras = lora_dict
    model.patch_lora_forward()
    pred = ask_mcq(model, tokenizer, task)

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pred": pred,
        "gold": task.gold_label,
        "correct": pred == task.gold_label,
        "num_sessions": len(task.sessions),
    }


def run_diagnostic(args):
    print(f"Loading model from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    # 如果是 CMP 模式，加载 gate checkpoint
    cmp_gate = None
    if args.mode == "cmp" and args.cmp_checkpoint:
        from cmp import CMPGate
        ckpt = torch.load(args.cmp_checkpoint, weights_only=False, map_location=model.device)
        cmp_gate = CMPGate(
            d_latent=ckpt.get("d_latent", 512),
            init_bias=ckpt.get("init_bias", -2.0),
        ).to(model.device)
        cmp_gate.load_state_dict(ckpt["gate_state_dict"])
        cmp_gate.eval()
        print(f"Loaded CMP gate from {args.cmp_checkpoint} "
              f"(epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc', '?')})")

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
        "standalone": evaluate_task_standalone,
        "rag": lambda model, tokenizer, task: evaluate_task_rag(
            model, tokenizer, task,
            top_k=args.rag_top_k, batch_size=args.rag_batch_size,
        ),
        "cmp": lambda model, tokenizer, task: evaluate_task_cmp(
            model, tokenizer, task, gate=cmp_gate,
        ),
    }

    type_names = {1: "Zero-Memory", 2: "In-Time", 3: "Post-Intervention"}

    def print_accuracy(results, label=""):
        if not results:
            return
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"  {label}accuracy: {acc:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
        for t in sorted(set(r["task_type"] for r in results)):
            group = [r for r in results if r["task_type"] == t]
            a = sum(r["correct"] for r in group) / len(group)
            print(f"    Type {t} ({type_names.get(t, '?')}): {a:.3f} ({sum(r['correct'] for r in group)}/{len(group)})")

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
            res["user_id"] = task.user_id
            results.append(res)
            model.reset()
            torch.cuda.empty_cache()
            status = "✓" if res["correct"] else "✗"
            print(f"  [{i+1}/{len(tasks)}] {status} task={task.task_id} type={task.task_type} pred={res['pred']} gold={res['gold']}")

        print(f"\n  --- Overall ---")
        print_accuracy(results)

        user_ids_in_results = sorted(set(r["user_id"] for r in results))
        if len(user_ids_in_results) > 1:
            for uid in user_ids_in_results:
                user_results = [r for r in results if r["user_id"] == uid]
                print(f"\n  --- User {uid} ---")
                print_accuracy(user_results)

        n_users = len(user_ids_in_results)
        out_path = os.path.join(args.output_dir, f"diagnostic_{mode}_{n_users}users.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "oracle", "single_shot", "naive_merge",
                                 "no_lora", "standalone", "rag", "cmp"])
    parser.add_argument("--cmp_checkpoint", type=str, default=None,
                        help="CMP gate checkpoint (.pt) for --mode cmp")
    parser.add_argument("--rag_top_k", type=int, default=10)
    parser.add_argument("--rag_batch_size", type=int, default=2)
    parser.add_argument("--max_users", type=int, default=2,
                        help="Max users to evaluate (0 = all)")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/perma/results")
    run_diagnostic(parser.parse_args())
