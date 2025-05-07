import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models import VisionTransformer, SwinTransformer


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def apply_lora(model, rank, alpha):
    # Add LoRA adapters to self-attention blocks (query, value)
    if isinstance(model, VisionTransformer):
        is_vit = True
    elif isinstance(model, SwinTransformer):
        is_vit = False
    else:
        raise NotImplementedError(f"Lora implemented only for timm VisionTransformer and SwinTransformer, got {type(model)}")
    assign_lora = partial(QkvWithLoRA, rank=rank, alpha=alpha)
    if is_vit:
        for block in model.blocks:
            block.attn.qkv = assign_lora(block.attn.qkv)
    else:
        for layer in model.layers:
            for block in layer.blocks:
                block.attn.qkv = assign_lora(block.attn.qkv)


    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA layers
    if is_vit:
        for block in model.blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_v.parameters():
                param.requires_grad = True
    else:
        for layer in model.layers:
            for block in layer.blocks:
                for param in block.attn.qkv.lora_q.parameters():
                    param.requires_grad = True
                for param in block.attn.qkv.lora_v.parameters():
                    param.requires_grad = True
