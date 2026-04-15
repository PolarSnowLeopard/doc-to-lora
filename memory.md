# Stateful HyperLoRA: Session-Level Recurrent Memory Parametrization for LLM Agents

## 1. 研究动机

### 1.1 核心问题

Agent 在多轮交互中需要维护长期记忆。当前工业界产品（Cursor、Claude、OpenClaw 等）全部采用文本级压缩（摘要/结构化存储），存在信息丢失不可逆、压缩后仍占上下文窗口等问题。

**核心假设**：将长期记忆编码进模型参数（LoRA），以外挂适配器的形式挂载到原始 LLM 上，可以在不损害模型原始性能的前提下实现更高效、更持久的记忆保持。

### 1.2 现有方案的 Gap

学术界已有超网络方案（Doc-to-LoRA、SHINE、GenerativeAdapter）可以将文本一次性编译为 LoRA 参数，但它们解决的是**静态文档**的参数化，而非**动态演化的 agent 记忆**。

| 属性 | 静态文档（现有工作的设定） | Agent Memory（真实场景） |
|------|-------------------------|------------------------|
| 时间性 | 一次性给定 | 持续增长，有时序依赖 |
| 可变性 | 不变 | 偏好会变、知识会更新 |
| 增量性 | 一次生成 LoRA | 需要增量更新，不能每次重新生成 |
| 异构性 | 同质的文本事实 | 事实/偏好/行为模式/工作流混杂 |
| 交互性 | 事实之间独立 | 记忆之间相互影响 |

**具体痛点**：现有超网络每次更新记忆必须将全部历史重新编译，计算量线性增长，且会超出超网络自身的输入长度限制。

### 1.3 我们的方案

将 agent 的长期记忆演化建模为**参数空间上的递推过程**（recurrence in parameter space），用一个 session-level recurrent hypernetwork 在低秩流形上做有状态的记忆更新。

```
现有工作：  f(context) → LoRA                              无状态，一次性
我们的方法：f(new_memory, state_t) → (LoRA_{t+1}, state_{t+1})   有状态，增量式
```

**与 RNN 的类比**：宏观结构类似 RNN，但在 session 级别而非 token 级别做递推。RNN 在 token 级别的缺陷（无法并行、梯度消失）在 session 级别不成立——session 天然顺序发生，序列长度仅几十到几百步。

**与 RNN 的本质区别**：

- 输出空间不同：输出的是低秩矩阵（Grassmann 流形），而非 token logits
- 闭环系统：LoRA 改变 LLM 行为 → 影响用户交互 → 影响下一轮产生什么新记忆

---

## 2. 相关工作

### 2.1 超网络生成 LoRA（推理时免梯度）

| 工作 | 会议/时间 | 核心做法 | 代码 |
|------|----------|---------|------|
| **GenerativeAdapter** | ICLR 2025 | 单次前向传播生成 LoRA，MSC 对话个性化任务验证 | `github.com/chentong0/generative-adapter` |
| **Doc-to-LoRA** | arXiv 2602.15902, 2026.02 | 超网络将文档编译为可热插拔的 LoRA 文件 | `github.com/SakanaAI/Doc-to-LoRA` |
| **SHINE** | arXiv 2602.06358, 2026.02 | 复用冻结 LLM 自身参数作为超网络骨干 | `github.com/Yewei-Liu/SHINE` |
| **CompAs** | 投稿 ICLR 2026 | 多个 adapter 可代数合并，支持可逆编码 | - |
| **HypeLoRA** | arXiv 2603.19278, 2026.03 | 超网络生成 LoRA A/B 矩阵，跨层结构耦合 | - |

### 2.2 LoRA 作为记忆的理论基础

| 工作 | 会议/时间 | 核心贡献 |
|------|----------|---------|
| **LoRA as Knowledge Memory** | 投稿 ICLR 2026, arXiv 2603.01097 | 首个系统性实证研究，分析 LoRA 存储容量、多模块组合、与 RAG/ICL 的协同 |
| **ParamMem** | arXiv 2602.23320, 2025.02 | 将反思模式编码进 LoRA 参数，反思多样性与任务成功率强正相关 (r=0.76) |
| **Sleeping LLM** | 2025 | 仿生 wake-sleep 循环，MEMIT(短期) → LoRA(长期) 渐进固化 |

### 2.3 Agent 多轮记忆管理

| 工作 | 时间 | 核心做法 |
|------|------|---------|
| **Focus** | arXiv 2601.07190, 2025.01 | Agent 自主决定何时压缩上下文，22.7% token 减少 |
| **Acon** | 2025.10 | 失败驱动的压缩策略优化，26-54% token 减少 |
| **Structured Distillation** | arXiv 2603.13017, 2026.03 | 对话压缩为 38 token 结构体，11x 压缩 |

### 2.4 Agent 记忆 Benchmark

| Benchmark | 时间 | 评测重点 |
|-----------|------|---------|
| **PERMA** | arXiv 2603.23231, 2026.03 | 跨 session 偏好演化、偏好一致性、跨领域记忆保持 |
| **MemoryArena** | arXiv 2602.16313, 2025.02 | 多 session agent 任务中的记忆获取和使用（766 tasks） |
| **MemoryCD** | 2026.03 | 真实用户跨年跨领域行为（Amazon Review 数据） |

---

## 3. Baselines 复现清单

### 3.1 超网络方案（直接对比对象）

| Baseline | 代码 | 复现方式 | 论文中角色 |
|----------|------|---------|-----------|
| **Doc-to-LoRA Oracle** | `github.com/SakanaAI/Doc-to-LoRA` | 预训练 checkpoint + 每次全量重编译 | 超网络上界 |
| **Doc-to-LoRA Single-shot** | 同上 | 每次只编译最新 session 的记忆 | 超网络下界 |
| **Doc-to-LoRA Naive Merge** | 同上 | 独立编译后 LoRA 参数加权平均 | Naive baseline |
| **SHINE** | `github.com/Yewei-Liu/SHINE` | 下载 checkpoint 跑 inference | 跨架构验证 |

### 3.2 传统记忆方案

| Baseline | 复现方式 | 论文中角色 |
|----------|---------|-----------|
| **Full Context** | 全部对话历史直接拼接塞入上下文 | 理论上界（受限于上下文长度） |
| **文本摘要** | LLM 总结压缩后塞入上下文 | 工业界主流做法 |
| **RAG** | FAISS + embedding model，检索 top-k 塞入上下文 | 工业界主流做法 |

### 3.3 论文实验表完整阵容

```
(a) Full Context                ← 理论上界
(b) 文本摘要                     ← 工业界 baseline
(c) RAG                         ← 工业界 baseline
(d) Doc-to-LoRA Oracle          ← 超网络上界（计算量线性增长）
(e) Doc-to-LoRA Single-shot     ← 超网络下界
(f) Doc-to-LoRA Naive Merge     ← naive 参数合并
(g) Stateful HyperLoRA（我们的方法）← 增量更新，GRU 门控融合
(h) ablation: 去掉 forget gate
(i) ablation: 去掉 update gate
```

---

## 4. 方法实现

### 4.1 基于 Doc-to-LoRA 的代码改动

**主改动文件**：`src/ctx_to_lora/modeling/hypernet.py`

```
SakanaAI/Doc-to-LoRA/
├── src/ctx_to_lora/modeling/
│   ├── hypernet.py          ← 主改动：HyperLoRA 加 state + gate
│   ├── aggregator.py        ← 不改
│   ├── ctx_encoder.py       ← 不改
│   ├── lora_layer.py        ← 不改
│   └── lora_merger.py       ← 可能小改（支持增量 merge 逻辑）
├── scripts/
│   └── our_exp/             ← 新建：训练和评估脚本
└── data/
    └── multi_session/       ← 新建：多 session 序列数据
```

**选 Doc-to-LoRA 的原因**：
- MIT 协议，Sakana AI 背书，667 stars，代码成熟度最高
- `internalize()` / `reset()` 接口干净，加 `update_memory()` 改动最小
- 预训练 checkpoint 直接可用（HuggingFace: `SakanaAI/doc-to-lora`），不需从头训超网络
- Gemma 基座，和 SHINE 的 Qwen3 互补可做跨架构验证

### 4.2 核心代码改动

#### 改动 1：HyperLoRA 加 state 和 gate

在 `HyperLoRA._init_model()` 末尾新增：

```python
self.state = None

gate_input_size = self.d_latent * 2
self.forget_gate = nn.Sequential(
    nn.Linear(gate_input_size, self.d_latent),
    nn.Sigmoid()
)
self.update_gate = nn.Sequential(
    nn.Linear(gate_input_size, self.d_latent),
    nn.Sigmoid()
)
self.state_proj = nn.Sequential(
    nn.Linear(gate_input_size, self.d_latent),
    nn.Tanh()
)
```

#### 改动 2：HyperLoRA.forward() 加入状态融合

```python
def forward(self, features, attn_mask=None, position_ids=None, n_ctx_chunks=None):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        # --- 新增：与旧状态融合 ---
        if self.state is not None:
            gate_input = torch.cat([lora_emb, self.state], dim=-1)
            f_gate = self.forget_gate(gate_input)
            u_gate = self.update_gate(gate_input)
            candidate = self.state_proj(gate_input)
            new_state = f_gate * self.state + u_gate * candidate
            lora_emb = new_state
        else:
            new_state = lora_emb.clone()

        self.state = new_state.detach()
        # --- 新增结束 ---

        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)

        return flat_loras, None
```

#### 改动 3：ModulatedPretrainedModel 加增量更新接口

```python
def update_memory(self, new_memory_str: str):
    """增量更新记忆，不重新编译全部历史"""
    ctx_tokenizer = get_tokenizer(self.ctx_encoder.base_model.name_or_path)
    ctx_ids = tokenize_ctx_text(
        dict(context=[new_memory_str]), ctx_tokenizer
    )["ctx_ids"]
    ctx_ids = torch.tensor(ctx_ids, device=self.device)
    ctx_attn_mask = torch.ones_like(ctx_ids)
    generated_loras, _ = self.generate_weights(ctx_ids, ctx_attn_mask)
    self.generated_loras = generated_loras

def reset(self):
    self.generated_loras = None
    self.hypernet.state = None  # 同时清除记忆状态
    # ... 原有 reset 逻辑 ...
```

#### 改后的使用方式

```python
# session 1: 首次编译
model.internalize("用户偏好Python, 缩进4空格")
model.generate(query_1)

# session 2: 增量更新（state 保留了 session 1 的记忆）
model.update_memory("用户开始使用TypeScript做前端")
model.generate(query_2)

# session N: forget gate 自动抑制过时偏好
model.update_memory("用户现在全部转向Rust")
model.generate(query_N)
```

### 4.3 训练 Pipeline

**训练数据格式**（多 session 序列）：

```json
{
    "sessions": [
        {"memory": "用户喜欢Python...", "qa": [["最喜欢的语言?", "Python"]]},
        {"memory": "用户开始学Rust...", "qa": [["最喜欢的语言?", "Python"], ["在学什么?", "Rust"]]},
        {"memory": "用户转向Rust...",   "qa": [["最喜欢的语言?", "Rust"], ["以前用什么?", "Python"]]}
    ]
}
```

**训练循环**：

```python
for sequence in dataset:
    model.reset()
    total_loss = 0

    for t, session in enumerate(sequence["sessions"]):
        if t == 0:
            model.internalize(session["memory"])
        else:
            model.update_memory(session["memory"])

        for q, a in session["qa"]:
            loss = model.forward(query=q, target=a)
            total_loss += loss

    total_loss.backward()
    optimizer.step()   # 只更新 gate/state_proj 参数，冻结原有超网络
```

**训练策略**：
- 第一阶段：冻结原有 Doc-to-LoRA 全部参数，只训练新增的 gate + state_proj（约几十万参数）
- 第二阶段（可选）：解冻 ResMLPBlock 联合微调

---

## 5. Benchmark 评测

### 5.1 主 Benchmark：PERMA

- **代码**：`github.com/PolarisLiu1/PERMA`，Apache 2.0
- **数据**：HuggingFace 数据集
- **评测内容**：跨 session 偏好演化、偏好一致性、跨领域记忆保持
- **评测协议**：多选题任务 + LLM-based 用户模拟器交互评测
- **选择理由**：专门评估偏好随时间变化，正好对应增量更新长期记忆的核心卖点

### 5.2 辅助 Benchmark：MemoryArena

- **数据**：HuggingFace Datasets（五个任务 split）
- **评测内容**：多 session agent 任务中的记忆获取和使用
- **选择理由**：比 PERMA 更侧重 action-level 记忆应用，验证跨任务类型通用性
- **注意**：代码仓库未完全公开，可能需自行编写评估脚本

### 5.3 自构造 Stress Test

- **目的**：测试极端场景（现有 benchmark 可能 session 数不够极端）
- **设计**：30+ session 长序列，包含偏好冲突 / 覆盖 / 遗忘场景
- **关键 case**：模拟用户偏好从 Python → TypeScript → Rust 的渐进迁移

---

## 6. 执行路线图

```
阶段 0: 诊断实验（~1-2 周，零代码改动）
├── clone Doc-to-LoRA，安装依赖，下载 checkpoint，验证 demo 能跑通
├── clone PERMA，下载数据集
├── 写评估脚本，将 PERMA 数据转成 Doc-to-LoRA 输入格式
├── 跑 Oracle / Single-shot / Naive merge 三组实验
└── 产出：motivation figure（确认 gap 存在，Oracle vs Single-shot 差距 >10%）

阶段 1: 核心代码改动（~2-3 周）
├── 在 hypernet.py 中实现 state + GRU gate
├── 实现 update_memory() 接口
├── 单元测试：验证增量更新的前向传播正确性
└── 产出：可用的 Stateful HyperLoRA 推理 pipeline

阶段 2: 训练 Pipeline（~3-4 周，最耗时）
├── 构造多 session 序列训练数据
├── 实现训练循环（冻结原有参数，只训 gate）
├── 训练 + 调参
└── 产出：训练好的 Stateful HyperLoRA checkpoint

阶段 3: 全量实验（~3-4 周）
├── PERMA 上跑全部 9 组对比 (a)-(i)
├── MemoryArena 上跑核心对比
├── Stress test 上跑 scaling 分析（记忆量 vs 性能 vs 计算量）
├── 消融实验（gate 组件的贡献）
├── 可视化：LoRA 参数空间演化轨迹 / gate 激活模式
└── 产出：完整实验数据和图表

阶段 4: 写论文（~3-4 周）
├── Introduction: motivation figure (阶段 0 产出)
├── Method: Stateful HyperLoRA 形式化定义 + 架构
├── Experiments: 全量对比 + 消融 + scaling 分析
├── Analysis: 可视化 + case study
└── 目标：NeurIPS 2026 (ddl ~2026.05) 或 ICML 2027 (ddl ~2027.01)

总计：~12-15 周
```

---

## 7. 预期论文结构

```
Title: "Stateful HyperLoRA: Session-Level Recurrent Memory Parametrization
        for Evolving LLM Agent Memory"

1. Introduction
   - Agent memory 的真实需求 vs 现有方案的 gap
   - Motivation figure: Oracle vs Single-shot 性能差距
   - 关键观察：静态编译 ≠ 动态演化

2. Related Work
   - Hypernetwork-based adapter generation
   - Agent memory management
   - Continual learning & memory consolidation

3. Problem Formulation
   - 形式化定义 Continual Memory Parametrization (CMP)
   - 与 continual learning / hypernetwork / RNN 的区别与联系

4. Method: Stateful HyperLoRA
   - 带隐状态的超网络架构
   - GRU 门控的记忆融合机制
   - 训练策略：冻结超网络 + 只训 gate

5. Experiments
   - PERMA / MemoryArena 上的全量对比 (9 组)
   - Stress test: 长序列 scaling 分析
   - 消融实验: gate 组件贡献
   - 计算效率对比: 增量更新 vs 全量重编译

6. Analysis
   - LoRA 参数空间的演化轨迹可视化
   - Gate 激活模式分析（何时遗忘、何时更新）
   - 失败案例分析

7. Conclusion
```

---

## 8. 关键参考文献

| 编号 | 论文 | arXiv / 会议 | 角色 |
|------|------|-------------|------|
| [1] | Doc-to-LoRA: Learning to Instantly Internalize Contexts | arXiv:2602.15902 | 主 baseline 代码基础 |
| [2] | SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA | arXiv:2602.06358 | 辅助 baseline |
| [3] | GenerativeAdapter: Contextualizing Language Models in Parameters with A Single Forward Pass | ICLR 2025 | 理论对标 |
| [4] | Understanding LoRA as Knowledge Memory: An Empirical Analysis | arXiv:2603.01097 | LoRA 记忆容量理论基础 |
| [5] | CompAs: Context Parametrization with Compositional Adapters | 投稿 ICLR 2026 | LoRA 可组合性 |
| [6] | PERMA: Benchmarking Personalized Memory Agents | arXiv:2603.23231 | 主 benchmark |
| [7] | MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks | arXiv:2602.16313 | 辅助 benchmark |
| [8] | ParamMem: Augmenting Language Agents with Parametric Reflective Memory | arXiv:2602.23320 | 参数化记忆相关工作 |
| [9] | Active Context Compression: Autonomous Memory Management in LLM Agents | arXiv:2601.07190 | Agent 记忆压缩相关工作 |
| [10] | On Catastrophic Forgetting in Low-Rank Decomposition-Based PEFT | arXiv:2603.09684 | LoRA 遗忘问题理论依据 |