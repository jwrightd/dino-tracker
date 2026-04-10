import torch
from torch import nn


class LoRAQKV(nn.Module):
    """Wrap a qkv linear layer and add LoRA updates to Q and V slices only."""

    def __init__(self, base_qkv: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError(f"Expected nn.Linear for qkv, got {type(base_qkv)!r}")

        self.base_qkv = base_qkv
        self.in_features = base_qkv.in_features
        self.out_features = base_qkv.out_features
        if self.out_features % 3 != 0:
            raise ValueError("Expected qkv out_features to be divisible by 3.")
        self.head_dim = self.out_features // 3
        self.rank = rank
        self.scale = alpha / float(rank)

        self.q_a = nn.Linear(self.in_features, rank, bias=False)
        self.q_b = nn.Linear(rank, self.head_dim, bias=False)
        self.v_a = nn.Linear(self.in_features, rank, bias=False)
        self.v_b = nn.Linear(rank, self.head_dim, bias=False)

        nn.init.kaiming_uniform_(self.q_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.q_b.weight)
        nn.init.kaiming_uniform_(self.v_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.v_b.weight)

        for param in self.base_qkv.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.base_qkv(x)
        q, k, v = base.split(self.head_dim, dim=-1)
        q = q + self.q_b(self.q_a(x)) * self.scale
        v = v + self.v_b(self.v_a(x)) * self.scale
        return torch.cat([q, k, v], dim=-1)


def inject_lora_qv(model: nn.Module, rank: int = 8, alpha: float = 16.0):
    for block in model.blocks:
        block.attn.qkv = LoRAQKV(block.attn.qkv, rank=rank, alpha=alpha)
    return model
