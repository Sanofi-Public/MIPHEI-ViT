"""
Removes foundation model encoder weights from a checkpoint, except for LoRA parameters.

Utility to prune foundation model weights from a PyTorch Lightning checkpoint,
keeping only generator (and LoRA) weights, and save in PyTorch and Safetensors formats.
"""

import sys
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.append("../")
from src.utils import get_generator_state_dict


def remove_foundation_model_ckpt(state_dict) -> dict:
    """
    Remove foundation model from a state dict checkpoint, except for LoRA parameters.

    This function iterates through the provided state dictionary and removes keys related to the
    foundation model's encoder (i.e., keys containing "generator.encoder.vit" or
    "generator.encoder.model"), unless the key also contains ".lora", in which case it is retained.
    All other keys are preserved. The idea here is to avoid sharing weights of excisting foundation
    model encoder, that can have restricted access.
    Args:
        state_dict (dict): The original state dictionary containing model parameters.
    Returns:
        dict: A new state dictionary with foundation model encoder parameters removed, except for
            LoRA parameters.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if ("generator.encoder.vit" in k) or ("generator.encoder.model" in k):
            if ".lora" in k:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove foundation model weights from checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path

    # Load and filter
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pruned_state_dict = remove_foundation_model_ckpt(ckpt["state_dict"])

    # Save standard PyTorch .ckpt
    ckpt["state_dict"] = pruned_state_dict
    new_ckpt_path = str(Path(ckpt_path).parent / "model_prune.weights.ckpt")
    torch.save(ckpt, new_ckpt_path)

    # Save Safetensors format
    pruned_safetensor_state_dict = get_generator_state_dict(pruned_state_dict)
    safetensor_path = str(Path(ckpt_path).parent / "model.safetensors")
    save_file(pruned_safetensor_state_dict, safetensor_path)
