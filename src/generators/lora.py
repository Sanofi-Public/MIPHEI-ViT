"""
Implementation of LoRA (Low-Rank Adaptation). Alternatively, the PEFT library can be used.

Paper: https://arxiv.org/abs/2106.09685
Code inspired from https://github.com/mnikitin/timm-vit-lora/blob/main/lora.py
"""

import torch
from functools import partial
from timm.models import VisionTransformer, SwinTransformer


class LoRALayer(torch.nn.Module):
    """
    Low-Rank Adaptation (LoRA) additonal layer for efficient fine-tuning of neural networks.

    This layer introduces trainable low-rank matrices A and B to adapt a pre-trained model
    with fewer parameters. The adaptation is controlled by the rank and scaling factor alpha.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        rank (int): Rank of the low-rank decomposition.
        alpha (float): Scaling factor for the adaptation.

    Attributes:
        A (torch.nn.Parameter): Learnable weight matrix of shape (in_dim, rank).
        B (torch.nn.Parameter): Learnable weight matrix of shape (rank, out_dim).
        alpha (float): Scaling factor for the adaptation.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (..., in_dim).

    Returns:
        torch.Tensor: Output tensor of shape (..., out_dim) after applying LoRA adaptation.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a scaled low-rank transformation to the input tensor x."""
        x = self.alpha * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    """
    QKV linear layer with LoRA (Low-Rank Adaptation) additional layer for the \
    query and value projections.

    This module wraps an existing QKV linear layer and injects LoRA-based low-rank updates into the
    query and value components, allowing for efficient fine-tuning.

    Args:
        qkv (torch.nn.Linear): The original QKV linear layer.
        rank (int): The rank of the LoRA low-rank decomposition.
        alpha (float): The scaling factor for the LoRA layers.

    Attributes:
        qkv (torch.nn.Linear): The wrapped QKV linear layer.
        dim (int): The input feature dimension.
        lora_q (LoRALayer): The LoRA layer applied to the query projection.
        lora_v (LoRALayer): The LoRA layer applied to the value projection.

    Methods:
        forward(x):
            Applies the QKV linear layer and adds the LoRA updates to the query and value
                components.

    """

    def __init__(self, qkv: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA modifications to the qkv projection and returns the updated tensor."""
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv


def apply_lora(model: torch.nn.Module, rank: int, alpha: float) -> None:
    """
    Apply Low-Rank Adaptation (LoRA) adapters to the self-attention blocks of a Vision Transformer \
    or Swin Transformer model.

    This function modifies the model in-place by:
    - Wrapping the query and value projection layers in each self-attention block with LoRA modules.
    - Freezing all model parameters except for the LoRA adapter parameters, which are set to be
        trainable.

    Args:
        model: The pretrained transformer model to which LoRA adapters will be applied.
            Must be an instance of VisionTransformer or SwinTransformer.
        rank (int): The rank (dimensionality) of the LoRA adapters.
        alpha (float): The scaling factor for the LoRA adapters.

    Raises:
        NotImplementedError: If the model is not a VisionTransformer or SwinTransformer.

    Returns:
        None
    """
    # Add LoRA adapters to self-attention blocks (query, value)
    if isinstance(model, VisionTransformer):
        is_vit = True
    elif isinstance(model, SwinTransformer):
        is_vit = False
    else:
        raise NotImplementedError(
            f"Lora implemented only for timm VisionTransformer and SwinTransformer, got "
            f"{type(model)}"
        )
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
