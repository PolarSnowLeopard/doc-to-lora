"""
CMP (Continual Memory Parametrization) — 递归门控 LoRA 记忆模块。

在 Doc-to-LoRA 的 Aggregator 输出（lora_emb）空间做门控合并，
将多 session 的信息增量融合为一个 LoRA 表示，再由冻结的 EinMix Head 生成 LoRA 参数。

Level 1: 线性门控（sigmoid interpolation）
Level 2: GRU 门控（TODO）
"""
import torch
import torch.nn as nn


class CMPGate(nn.Module):
    """Level 1: 线性门控合并。

    z = sigmoid(W @ [h_prev, q_new] + b)
    h_new = z * h_prev + (1-z) * q_new

    操作维度: d_latent (512), 在所有层和 rank 间共享权重。
    """

    def __init__(self, d_latent: int = 512, init_bias: float = -2.0):
        super().__init__()
        self.gate = nn.Linear(d_latent * 2, d_latent)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, h_prev, q_new):
        """
        h_prev, q_new: [*, d_latent]  (任意前导维度，如 [32, 1, 8, 512])
        returns: h_new  [*, d_latent]
        """
        z = torch.sigmoid(self.gate(torch.cat([h_prev, q_new], dim=-1)))
        return z * h_prev + (1 - z) * q_new


def extract_aggregator_output(model, text: str, max_tokens: int = 4000):
    """从冻结的 D2L 模型中提取单个 session 的 aggregator 输出 (lora_emb)。

    Returns: lora_emb [1, n_layers, n_modules, r, d_latent]  (e.g. [1, 32, 1, 8, 512])
    """
    from ctx_to_lora.model_loading import get_tokenizer

    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    tokens = ctx_tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    ctx_ids = torch.tensor([tokens], device=model.device)
    ctx_attn_mask = torch.ones_like(ctx_ids)

    with torch.no_grad():
        ctx_features = model.ctx_encoder(
            input_ids=ctx_ids, attention_mask=ctx_attn_mask
        )

    with torch.no_grad():
        lora_emb, _ = model.hypernet.aggregator(ctx_features, ctx_attn_mask, None)

    return lora_emb  # [1, 32, 1, 8, 512]


def lora_emb_to_lora_dict(hypernet, lora_emb):
    """将 lora_emb 通过冻结的 ResMLPBlock + L2Norm + EinMix Head 转为 LoRA 权重字典。

    lora_emb: [bs, n_layers, n_modules, r, d_latent]
    returns: lora_dict  {module_name: {"A": tensor, "B": tensor}}
    """
    h = hypernet.layers(lora_emb)
    norm = torch.norm(h, dim=-1, keepdim=True)
    h = h / norm
    flat_loras = hypernet.head(h)
    return hypernet._to_lora_dict(flat_loras)


def run_cmp_sessions(gate: CMPGate, lora_embs: list[torch.Tensor]) -> torch.Tensor:
    """将一系列 session 的 lora_emb 通过 CMP gate 递归合并。

    lora_embs: list of [1, 32, 1, 8, 512] tensors
    returns: h  [1, 32, 1, 8, 512]  — 最终的合并状态
    """
    h = torch.zeros_like(lora_embs[0])
    for emb in lora_embs:
        h = gate(h, emb)
    return h
