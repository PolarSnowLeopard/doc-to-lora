"""快速三组对比测试：验证 base model 能力 & Doc-to-LoRA OOD 问题"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from data_adapter import load_tasks, session_to_text, ALL_USER_IDS

ckpt = sys.argv[1] if len(sys.argv) > 1 else "trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"
sd = torch.load(ckpt, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(sd, train=False, use_sequence_packing=False)
model.reset()
tok = get_tokenizer(model.base_model.name_or_path)

MAX_INPUT = 7500

def build_prompt_and_ids(question, options, ctx=None):
    n = len(options)
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    if ctx:
        content = f"Based on the conversation below, answer the question.\n\nConversation:\n{ctx}\n\nQuestion: {question}\n\nOptions:\n{opts}\n\nAnswer with the letter only (A-{chr(65+n-1)}):"
    else:
        content = f"Question: {question}\n\nOptions:\n{opts}\n\nAnswer with the letter only (A-{chr(65+n-1)}):"
    chat = [{"role": "user", "content": content}]
    ids = tok.apply_chat_template(chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
    return ids

def generate_and_print(input_ids, use_base=False):
    gen_model = model.base_model if use_base else model
    with torch.inference_mode():
        out = gen_model.generate(input_ids=input_ids, max_new_tokens=32, pad_token_id=tok.eos_token_id)
    new_tok = out[0][input_ids.shape[-1]:]
    raw = tok.decode(new_tok, skip_special_tokens=False)
    clean = tok.decode(new_tok, skip_special_tokens=True)
    print(f"  raw:   {raw}")
    print(f"  clean: {clean}")
    return clean

# ── Test 1: 纯 MCQ，无上下文 ──
print("=" * 60)
print("Test 1: 纯 MCQ (无上下文, 无 LoRA)")
print("=" * 60)
ids = build_prompt_and_ids("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"])
print(f"  input_len: {ids.shape[-1]}")
generate_and_print(ids, use_base=True)

# ── 加载 PERMA 任务并检查选项 ──
tasks = load_tasks(user_ids=[ALL_USER_IDS[0]], noise=False, multi_domain=False)
print(f"\n加载了 {len(tasks)} 个任务")
for i, t in enumerate(tasks[:5]):
    prompt_ids = build_prompt_and_ids(t.question, t.options)
    print(f"  [{i}] {t.task_id} type={t.task_type} | {len(t.options)} options | prompt_tokens={prompt_ids.shape[-1]} | gold={t.gold_label}")

task = tasks[0]

# ── Test 2: PERMA MCQ, 上下文在 prompt, 无 LoRA ──
print("\n" + "=" * 60)
print(f"Test 2: PERMA MCQ (上下文在 prompt, 无 LoRA) — {task.task_id}")
print("=" * 60)
prompt_ids = build_prompt_and_ids(task.question, task.options)
prompt_len = prompt_ids.shape[-1]
ctx_budget = MAX_INPUT - prompt_len - 100
ctx_full = session_to_text(task.sessions[-1])
ctx_tokens = tok.encode(ctx_full, add_special_tokens=False)[:max(ctx_budget, 300)]
ctx = tok.decode(ctx_tokens, skip_special_tokens=True)

ids = build_prompt_and_ids(task.question, task.options, ctx=ctx)
print(f"  input_len: {ids.shape[-1]} (prompt_only: {prompt_len}, ctx_tokens: {len(ctx_tokens)})")
print(f"  gold: {task.gold_label}")
model.reset()
pred2 = generate_and_print(ids, use_base=True)

# ── Test 3: PERMA MCQ, internalize + LoRA ──
print("\n" + "=" * 60)
print(f"Test 3: PERMA MCQ (internalize + LoRA) — {task.task_id}")
print("=" * 60)
model.reset()
ctx_for_intern = tok.encode(ctx_full, add_special_tokens=False)[:4000]
ctx_intern_text = tok.decode(ctx_for_intern, skip_special_tokens=True)
model.internalize(ctx_intern_text)
ids = build_prompt_and_ids(task.question, task.options)
print(f"  input_len: {ids.shape[-1]}")
print(f"  gold: {task.gold_label}")
pred3 = generate_and_print(ids, use_base=False)

# ── Test 4: Doc-to-LoRA 原始用途验证（文档 QA）──
print("\n" + "=" * 60)
print("Test 4: Doc-to-LoRA 文档 QA (正常用途)")
print("=" * 60)
model.reset()
torch.cuda.empty_cache()
doc = "Sakana AI is a company based in Tokyo Japan founded in 2023. It focuses on nature-inspired AI research."
model.internalize(doc)
ids = build_prompt_and_ids("Where is Sakana AI based?", ["London", "Tokyo", "Beijing", "New York"])
print(f"  input_len: {ids.shape[-1]}")
pred4 = generate_and_print(ids, use_base=False)
print(f"  gold: B")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"  Test 1 (纯 MCQ):           模型能力正常")
print(f"  Test 2 (PERMA+context):    pred={pred2.strip()[:30]} gold={task.gold_label}")
print(f"  Test 3 (PERMA+LoRA):       pred={pred3.strip()[:30]} gold={task.gold_label}")
print(f"  Test 4 (Doc QA+LoRA):      pred={pred4.strip()[:30]} gold=B")
