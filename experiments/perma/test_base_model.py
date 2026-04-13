"""快速测试：不加 LoRA，验证 Mistral-7B 能否正常回答 MCQ"""
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

# Test 1: 纯 MCQ，无上下文，不 internalize
print("=" * 60)
print("Test 1: 纯 MCQ (无上下文, 无 LoRA)")
print("=" * 60)
chat = [{"role": "user", "content": "Question: What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid\nAnswer with the letter only:"}]
ids = tok.apply_chat_template(chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(input_ids=ids, max_new_tokens=16, pad_token_id=tok.eos_token_id)
new_tok = out[0][ids.shape[-1]:]
print(f"  raw: {tok.decode(new_tok, skip_special_tokens=False)}")
print(f"  clean: {tok.decode(new_tok, skip_special_tokens=True)}")

# Test 2: PERMA MCQ，上下文放 prompt，不 internalize
print("\n" + "=" * 60)
print("Test 2: PERMA MCQ (上下文在 prompt, 无 LoRA)")
print("=" * 60)
tasks = load_tasks(user_ids=[ALL_USER_IDS[0]], noise=False, multi_domain=False)
task = tasks[0]
ctx = session_to_text(task.sessions[-1])[:2000]
n = len(task.options)
opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(task.options))
content = f"Based on the conversation below, answer the question.\n\nConversation:\n{ctx}\n\nQuestion: {task.question}\n\nOptions:\n{opts}\n\nAnswer with the letter only (A-{chr(65+n-1)}):"
chat = [{"role": "user", "content": content}]
ids = tok.apply_chat_template(chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(f"  input_len: {ids.shape[-1]}")
with torch.inference_mode():
    out = model.generate(input_ids=ids, max_new_tokens=16, pad_token_id=tok.eos_token_id)
new_tok = out[0][ids.shape[-1]:]
print(f"  raw: {tok.decode(new_tok, skip_special_tokens=False)}")
print(f"  clean: {tok.decode(new_tok, skip_special_tokens=True)}")
print(f"  gold: {task.gold_label}")

# Test 3: PERMA MCQ，internalize 后生成
print("\n" + "=" * 60)
print("Test 3: PERMA MCQ (internalize + LoRA)")
print("=" * 60)
model.reset()
model.internalize(ctx)
chat = [{"role": "user", "content": f"Question: {task.question}\n\nOptions:\n{opts}\n\nAnswer with the letter only (A-{chr(65+n-1)}):"}]
ids = tok.apply_chat_template(chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(input_ids=ids, max_new_tokens=16, pad_token_id=tok.eos_token_id)
new_tok = out[0][ids.shape[-1]:]
print(f"  raw: {tok.decode(new_tok, skip_special_tokens=False)}")
print(f"  clean: {tok.decode(new_tok, skip_special_tokens=True)}")
print(f"  gold: {task.gold_label}")
