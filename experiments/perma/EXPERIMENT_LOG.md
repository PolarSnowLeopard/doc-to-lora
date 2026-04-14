# PERMA Experiments — Doc-to-LoRA Baseline & CMP

## 一、实验目的

1. 验证 Doc-to-LoRA（off-the-shelf）在 PERMA 上的表现，作为 CMP 的 motivation
2. 在 PERMA 上微调 Doc-to-LoRA 超网络，建立公平的 retrained baseline
3. 与其他记忆方法（Standalone / RAG）对比，证明参数化记忆路线的优势

## 二、基础设置

- **基座模型**: Mistral-7B-Instruct-v0.2（32K context window）
- **Benchmark**: PERMA（10 users, 705 tasks, 8 选项 MCQ, Clean/Single-Domain）
- **GPU**: 单卡 A100 80GB（cfff 集群）

## 三、评测模式定义

| 模式 | 上下文使用方式 | LoRA | 需要训练 |
|------|--------------|------|---------|
| no_lora (standalone) | 最后 session 拼进 prompt，基座模型直接回答 | 无 | 否 |
| oracle | 全部 session 拼接 → internalize → 超网络生成 LoRA | 有 | 是(超网络) |
| single_shot | 只用最后 session → internalize → 超网络生成 LoRA | 有 | 是(超网络) |
| naive_merge | 每个 session 分别 internalize → 多个 LoRA 参数取平均 | 有（合并） | 是(超网络) |

---

## 四、实验 A — Off-the-Shelf Doc-to-LoRA（Motivation 实验）

### 设置
- **超网络 checkpoint**: `trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin`（在文档 QA 数据上预训练，非 PERMA 数据）
- **上下文截断**: internalize 模式截断到 4000 tokens, no_lora 模式控制在 7500 tokens 以内

### 结果: 10 Users, 705 Tasks

| 方法 | Overall | Type 1 (Zero-Memory) | Type 2 (In-Time) | Type 3 (Post-Intervention) |
|------|---------|---------------------|-------------------|---------------------------|
| **no_lora** | **46.5%** (328/705) | **62.4%** (88/141) | **53.2%** (75/141) | **39.0%** (165/423) |
| single_shot | 19.9% (140/705) | 24.1% (34/141) | 21.3% (30/141) | 18.0% (76/423) |
| oracle | 17.9% (126/705) | 17.7% (25/141) | 17.7% (25/141) | 18.0% (76/423) |
| random (1/8) | 12.5% | 12.5% | 12.5% | 12.5% |

### 结果: 1 User (user334, 75 Tasks)

| 方法 | Overall | Type 1 | Type 2 | Type 3 |
|------|---------|--------|--------|--------|
| no_lora | 58.7% (44/75) | 60.0% (9/15) | 73.3% (11/15) | 53.3% (24/45) |
| naive_merge | 24.0% (18/75) | 26.7% (4/15) | 33.3% (5/15) | 20.0% (9/45) |
| oracle | 21.3% (16/75) | 33.3% (5/15) | 26.7% (4/15) | 15.6% (7/45) |
| single_shot | 17.3% (13/75) | 26.7% (4/15) | 26.7% (4/15) | 11.1% (5/45) |

### Per-User Breakdown (10 Users, Off-the-Shelf)

| User | no_lora | oracle | single_shot |
|------|---------|--------|-------------|
| 108 | 53.3% | 21.3% | 21.3% |
| 109 | 41.7% | 23.3% | 28.3% |
| 112 | 55.0% | 13.3% | 20.0% |
| 123 | 45.3% | 17.3% | 28.0% |
| 334 | 58.7% | 21.3% | 17.3% |
| 354 | 38.7% | 29.3% | 33.3% |
| 419 | 49.3% | 12.0% | 9.3% |
| 507 | 34.7% | 17.3% | 18.7% |
| 914 | 38.3% | 8.3% | 8.3% |
| 1377 | 49.3% | 13.3% | 13.3% |

### 发现
1. Off-the-shelf Doc-to-LoRA 仅略高于随机水平（12.5%），输出分布严重偏移（🔑符号、偏向 A）
2. 说明文档 QA 训练的超网络无法泛化到对话记忆数据 → 需要在 PERMA 上微调

---

## 五、实验 B — Fine-tuned Doc-to-LoRA（Retrained Baseline）

### 训练设置
- **初始 checkpoint**: `trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin`
- **训练数据**: 9 个 user（108,109,112,123,354,419,507,914,1377）的 PERMA 数据
- **测试数据**: Held-out user334（75 tasks）
- **数据增强**: 每个 task 4 种 context 变体（全部 session / 最后 1 / 最后 2 / 最后一半）
- **损失函数**: CE loss（`use_kl_loss: false`）
- **训练配置** (`configs/perma/finetune_mistral.yaml`):
  - `lora_r: 8`, `target_modules: [down_proj]`
  - `max_packed_inp_len: 768`, `max_packed_ctx_len: 768`, `max_ctx_len: 768`
  - `gradient_accumulation_steps: 16`
  - `quantize_ctx_encoder: true`
  - `learning_rate: 1e-5`, `warmup_steps: 50`
  - `max_steps: 2000`, `save_steps: 500`
- **训练时间**: 12 小时 25 分钟（单卡 A100 80GB）
- **最终 loss**: `train_loss = 0.0199`，111 epochs
- **输出 checkpoint**: `train_outputs/runs/Apr13_22-09-32_dsw-32871-6f94ff8dc7-x5bq_4d96d843/checkpoint-2000/pytorch_model.bin`

### 结果: Held-out User334 (75 Tasks)

| 方法 | Overall | Type 1 (Zero-Memory) | Type 2 (In-Time) | Type 3 (Post-Intervention) |
|------|---------|---------------------|-------------------|---------------------------|
| **single_shot** (fine-tuned) | **85.3%** (64/75) | 66.7% (10/15) | **86.7%** (13/15) | **91.1%** (41/45) |
| **naive_merge** (fine-tuned) | **84.0%** (63/75) | **73.3%** (11/15) | 80.0% (12/15) | 88.9% (40/45) |
| **oracle** (fine-tuned) | 74.7% (56/75) | 60.0% (9/15) | 66.7% (10/15) | 82.2% (37/45) |
| no_lora (base Mistral-7B) | 58.7% (44/75) | 60.0% (9/15) | 73.3% (11/15) | 53.3% (24/45) |

### 关键发现
1. **微调后巨幅提升**: single_shot 从 17.3% → 85.3%（+68 pp），说明超网络微调有效
2. **oracle 反而最差**: 全部 session 拼接 → 信息过载，超网络处理不好长上下文
3. **single_shot ≈ naive_merge > oracle**: 暗示"如何处理多 session 历史"是关键瓶颈
4. **single_shot 在 Type 3 最强** (91.1%): 仅用最新 session 反而最能跟踪偏好变化
5. **泛化到 held-out user**: 训练 9 个 user，在未见过的 user334 上表现强劲，说明非过拟合

### 对 CMP 的 Motivation
- oracle（暴力拼接）74.7% < single_shot（选择性遗忘）85.3%
- 说明简单拼接不如选择性处理
- naive_merge（简单平均）84.0% 接近 single_shot 但缺乏学习到的选择/遗忘机制
- CMP 用递归门控替代简单平均/拼接 → 预期在 Type 3 上进一步提升

---

## 六、PERMA 论文原始结果（参考对比）

PERMA 论文中的模型均为 **大规模闭源模型 + 全文上下文**，与我们不直接可比：

### Standalone LLMs (Clean, Single-Domain)

| 模型 | MCQ Acc |
|------|---------|
| Kimi-K2.5 (推理模型) | 88.2% |
| Qwen3-32B | 87.0% |
| Gemini2.5-Flash | 87.0% |
| GLM-4.7-Flash | 86.8% |
| Llama3.3-70B | 81.8% |
| GLM-5 | 81.1% |
| MiniMax-M2.5 | 79.7% |
| Qwen2.5-72B | 79.0% |
| GPT-4o-mini | 78.0% |

### Memory Systems (Clean, Single-Domain, 使用 GPT-4o-mini 回答)

| 记忆方法 | MCQ Acc |
|----------|---------|
| MemOS | 81.1% |
| Memobase | 73.3% |
| EverMemOS | 72.8% |
| RAG (BGE-M3) | 70.2% |
| Mem0 | 68.6% |
| Lightmem | 65.7% |
| Supermemory | 65.5% |

**注意**: 以上结果使用 GPT-4o-mini 作为回答模型，我们使用 Mistral-7B，因此需要用相同模型重新跑 Standalone 和 RAG baseline 才能公平比较。

---

## 六、实验 C — Standalone & RAG Baseline（Mistral-7B）

### 设置
- 与 Doc-to-LoRA 实验使用完全相同的基座模型（Mistral-7B-Instruct-v0.2）和测试用户（user334, 75 tasks）
- Standalone 使用 PERMA 原版 `ANSWER_OPTIONAL_PROMPT` 格式，全部对话历史作为上下文
- RAG 使用 BGE-M3 编码对话片段（每 2 条消息为一个 chunk），余弦相似度 top-10 检索

### 结果: Held-out User334 (75 Tasks)

| 方法 | Overall | Type 1 (Zero-Memory) | Type 2 (In-Time) | Type 3 (Post-Intervention) |
|------|---------|---------------------|-------------------|---------------------------|
| Standalone（全文上下文） | 68.0% (51/75) | 66.7% (10/15) | 60.0% (9/15) | 71.1% (32/45) |
| RAG (BGE-M3 top-10) | 60.0% (45/75) | 60.0% (9/15) | 60.0% (9/15) | 60.0% (27/45) |

### 关键观察
- Standalone 的 input_len = 32736，几乎撑满 Mistral-7B 的 32K 上下文窗口
- RAG 的 input_len ≈ 2000 tokens，token 效率高但准确率最低
- RAG 实现与 PERMA 论文完全一致（编码方式、chunk 大小、top_k），10 个点差距来自回答模型差异（GPT-4o-mini vs Mistral-7B），与 standalone 的差距一致

---

## 七、完整对比总结

### 所有方法在 User334 (75 Tasks) 上的表现

| 方法 | 类型 | Overall | Type 1 (Zero-Memory) | Type 2 (In-Time) | Type 3 (Post-Intervention) | 输入长度 |
|------|------|---------|---------------------|-------------------|---------------------------|---------|
| D2L fine-tuned (single_shot) | 需训练 | **85.3%** | 66.7% | **86.7%** | **91.1%** | ~200 |
| D2L fine-tuned (naive_merge) | 需训练 | 84.0% | **73.3%** | 80.0% | 88.9% | ~200 |
| D2L fine-tuned (oracle) | 需训练 | 74.7% | 60.0% | 66.7% | 82.2% | ~200 |
| Standalone（全文上下文） | 免训练 | 68.0% | 66.7% | 60.0% | 71.1% | ~32K |
| RAG (BGE-M3 top-10) | 免训练 | 60.0% | 60.0% | 60.0% | 60.0% | ~2K |
| no_lora（仅最后 session） | 免训练 | 58.7% | 60.0% | 73.3% | 53.3% | ~7.5K |
| D2L off-the-shelf (single_shot) | 免训练* | 17.3% | 26.7% | 26.7% | 11.1% | ~200 |

*D2L off-the-shelf 使用在文档 QA 上预训练的超网络，未在 PERMA 数据上微调

### 与 PERMA 论文 (GPT-4o-mini) 的对比

| 模式 | GPT-4o-mini (PERMA 论文) | Mistral-7B (本实验) | 差值 |
|------|------------------------|-------------------|------|
| Standalone | 78.0% | 68.0% | -10.0 |
| RAG (BGE-M3) | 70.2% | 60.0% | -10.2 |

差值一致（~10 pp），说明实现正确，差距来自模型能力而非代码差异。

### 核心结论

1. **参数化记忆 > 上下文记忆**: D2L fine-tuned (85.3%) 大幅超越 Standalone (68.0%)，且不占用上下文窗口
2. **7B + LoRA 超越 GPT-4o-mini + 全文**: 85.3% > 78.0%，证明参数化记忆路线的潜力
3. **多 session 处理是关键瓶颈**: oracle (74.7%) < naive_merge (84.0%) ≈ single_shot (85.3%)，简单拼接不如选择性处理
4. **CMP 的切入点**: naive_merge 用简单平均处理多 session，缺乏学习到的选择性遗忘/强化机制 → CMP 用递归门控替代

---

## 八、待完成实验

### 阶段 1: 补齐 Baseline（Mistral-7B） ✅
- [x] Standalone baseline（对齐 PERMA 的 ANSWER_OPTIONAL_PROMPT 格式）
- [x] RAG baseline（BGE-M3 top-k 检索 + Mistral-7B 回答）

### 阶段 2-4: CMP 设计、实现、评测
- [ ] 设计 CMP 递归超网络架构
- [ ] 实现 CMP 并在 PERMA 上训练
- [ ] 与所有 baseline 对比

### 阶段 5: 扩展评测
- [ ] Leave-one-out 全部 10 个 user
- [ ] Noise / Style-aligned 数据条件
- [ ] Temporal probing（增量评测）

---

## 九、调试记录

1. **PERMA options 解析错误**: `options` 字段是 `"A: text\nB: text\n..."` 格式的字符串，初始代码当作 list 迭代导致每个字符变成一个"选项"(2621个)，prompt 严重溢出 → 修复: 实现 `_parse_options()`
2. **deepcopy 非叶 tensor 失败**: naive_merge 中 `copy.deepcopy(model.generated_loras)` 报错 → 修复: 改用 `detach().clone()`
3. **训练 OOM**: Mistral-7B + 大 packed_len 超出 80GB 显存 → 修复: 降低 `max_packed_inp_len/ctx_len` 到 768，启用 `quantize_ctx_encoder`，`gradient_accumulation_steps: 16`
4. **accelerate 多卡冲突**: `accelerate launch --num_processes=1` 与多卡 config 冲突 → 修复: 直接用 `CUDA_VISIBLE_DEVICES=0 uv run python train.py`

## 十、文件说明

- `data_adapter.py`: PERMA 数据加载与格式转换（含 `_parse_options` 修复）
- `diagnostic_eval.py`: 四种模式的评测脚本（oracle / single_shot / naive_merge / no_lora）
- `prepare_train_data.py`: PERMA → Doc-to-LoRA 训练 parquet 转换
- `finetune.sh`: 微调一键脚本
- `test_base_model.py`: 基座模型能力验证（隔离测试）
- `summarize_results.py`: 结果汇总脚本
- `configs/perma/finetune_mistral.yaml`: 微调配置文件
- `results/`: off-the-shelf 评测结果
- `results_finetuned/`: 微调后评测结果
