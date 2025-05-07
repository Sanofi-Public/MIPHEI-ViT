import argparse
from pathlib import Path
from safetensors.torch import save_file
import torch


def remove_foundation_model_ckpt(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if ("generator.encoder.vit" in k) or ("generator.encoder.model" in k):
            if ".lora" in k:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_generator_state_dict(state_dict):
    generator_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            generator_state_dict[k.replace("generator.", "")] = v
    return generator_state_dict


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
