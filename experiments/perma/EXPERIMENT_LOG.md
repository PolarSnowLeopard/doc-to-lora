# PERMA Diagnostic Experiments — Doc-to-LoRA Off-the-Shelf Evaluation

## 实验目的

验证 Doc-to-LoRA（在文档 QA 上训练的超网络）直接应用于 PERMA 对话记忆 benchmark 的效果，作为后续 CMP (Continual Memory Parametrization) 方案的 motivation 实验。

## 实验设置

- **基座模型**: Mistral-7B-Instruct-v0.2
- **超网络 checkpoint**: `trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin`（Doc-to-LoRA 在文档 QA 数据上训练）
- **Benchmark**: PERMA（10 users, 705 tasks, 8 选项 MCQ）
- **上下文截断**: internalize 模式截断到 4000 tokens, no_lora 模式动态计算 budget 控制在 7500 tokens 以内
- **GPU**: 单卡 A100 80GB

## 四种评测模式

| 模式 | 上下文使用方式 | LoRA |
|------|--------------|------|
| no_lora | 最后 session 拼进 prompt，基座模型直接回答 | 无 |
| oracle | 全部 session 拼接 → internalize → 超网络生成 LoRA | 有 |
| single_shot | 只用最后 session → internalize → 超网络生成 LoRA | 有 |
| naive_merge | 每个 session 分别 internalize → 多个 LoRA 参数取平均 | 有（合并） |

## 主要结果

### 10 Users, 705 Tasks

| 方法 | Overall | Type 1 (Zero-Memory) | Type 2 (In-Time) | Type 3 (Post-Intervention) |
|------|---------|---------------------|-------------------|---------------------------|
| **no_lora** | **46.5%** (328/705) | **62.4%** (88/141) | **53.2%** (75/141) | **39.0%** (165/423) |
| single_shot | 19.9% (140/705) | 24.1% (34/141) | 21.3% (30/141) | 18.0% (76/423) |
| oracle | 17.9% (126/705) | 17.7% (25/141) | 17.7% (25/141) | 18.0% (76/423) |
| random (1/8) | 12.5% | 12.5% | 12.5% | 12.5% |

### 1 User (user334, 75 Tasks) — 含 naive_merge

| 方法 | Overall | Type 1 | Type 2 | Type 3 |
|------|---------|--------|--------|--------|
| no_lora | 58.7% (44/75) | 60.0% (9/15) | 73.3% (11/15) | 53.3% (24/45) |
| naive_merge | 24.0% (18/75) | 26.7% (4/15) | 33.3% (5/15) | 20.0% (9/45) |
| oracle | 21.3% (16/75) | 33.3% (5/15) | 26.7% (4/15) | 15.6% (7/45) |
| single_shot | 17.3% (13/75) | 26.7% (4/15) | 26.7% (4/15) | 11.1% (5/45) |

### Per-User Breakdown (10 Users)

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

## 关键发现

1. **In-context learning 远超 Doc-to-LoRA 所有变体**: 46.5% vs ~18-20%，差距在所有 user 上一致
2. **Oracle 反而不如 single_shot** (17.9% < 19.9%): 拼接全部 session 后截断到 4000 tokens 反而丢失了关键信息
3. **Doc-to-LoRA 变体仅略高于随机水平** (12.5%): 超网络生成的 LoRA 基本没有有效编码对话记忆
4. **LoRA 输出分布偏移**: 生成文本常以 🔑/🔔 等特殊符号开头，并强烈偏向选项 A，说明文档训练的 LoRA 在对话数据上产生了系统性偏差
5. **Type 3 (偏好演变) 最难**: 所有方法在此类型上表现最差，反映了跟踪偏好变化的难度
6. **Per-user 方差大**: no_lora 从 34.7% 到 58.7%，oracle 从 8.3% 到 29.3%

## 调试过程中发现的 Bug

1. **PERMA options 解析错误**: `options` 字段是 `"A: text\nB: text\n..."` 格式的字符串，初始代码当作 list 迭代导致每个字符变成一个"选项"(2621个)，prompt 严重溢出产生乱码
2. **deepcopy 非叶 tensor 失败**: naive_merge 中 `copy.deepcopy(model.generated_loras)` 对计算图中的 tensor 报错，改用 `detach().clone()`
3. **上下文截断不准确**: 需要考虑 chat template 和选项文本本身的 token 开销，动态计算 context budget

## 结论与下一步

当前结果确认 off-the-shelf Doc-to-LoRA 不能直接用于对话记忆任务。**CMP 要弥合的 gap = 46.5% (in-context) - 18% (Doc-to-LoRA) ≈ 28 个百分点**。

下一步：
1. 将 PERMA 数据适配到 Doc-to-LoRA 训练格式，在 PERMA 上微调超网络（公平 baseline）
2. 设计并实现 CMP 递归超网络架构
3. 在同等训练条件下对比 CMP vs retrained Doc-to-LoRA vs no_lora

## 文件说明

- `data_adapter.py`: PERMA 数据加载与格式转换
- `diagnostic_eval.py`: 四种模式的评测脚本
- `test_base_model.py`: 基座模型能力验证（隔离测试）
- `summarize_results.py`: 结果汇总脚本
- `results/`: 评测结果 JSON 文件
