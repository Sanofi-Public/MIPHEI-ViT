from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

from benchmark.models.pix2pix import UnetGenerator
from src.generators import get_generator
from src.generators.hemit_models import ResnetGeneratorSwinT, resize_embed_hemit_statedict
from src.utils import validate_load_info, get_generator_state_dict
from benchmark.models.rosie import get_model as get_rosie_model
from benchmark.models.diffusionft import load_diffusion_pipeline
from benchmark.evaluators.utils import adapt_checkpoint_hemit



def get_pix2pix(checkpoint_dir: Path, cfg_model: OmegaConf, device: torch.device) -> torch.nn.Module:
    output_nc = len(cfg_model.data.targ_channel_names)

    torch.cuda.empty_cache()
    model = UnetGenerator(
        3, output_nc, 8, ngf=64,
        norm_layer=torch.nn.BatchNorm2d,
        use_dropout=False,
    )
    state_dict = torch.load(checkpoint_dir / "best.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_miphei(checkpoint_dir: Path, cfg_model: OmegaConf, device: torch.device, H: int, W: int) -> torch.nn.Module:

    torch.cuda.empty_cache()
    nc_out = len(cfg_model.data.targ_channel_names)
    nc_in, width, _ = 3, H, W

    model = get_generator(
        cfg_model.model.model_name, width, nc_in, nc_out, cfg_model
    )

    use_safetensors = (checkpoint_dir / "model.safetensors").exists()
    if use_safetensors:
        from safetensors.torch import load_file
        checkpoint_path = str(checkpoint_dir / "model.safetensors")
        state_dict = load_file(checkpoint_path, device="cpu")
        strict_load = False
        print(f"[MIPHEI] Loading checkpoint from safetensors: {checkpoint_path}")
    else:
        checkpoint_path = str(checkpoint_dir / "model.weights.ckpt")
        ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = get_generator_state_dict(ckpt)
        strict_load = True
        print(f"[MIPHEI] Loading checkpoint from ckpt: {checkpoint_path}")

    if hasattr(model, "swinT"):
        state_dict = resize_embed_hemit_statedict(state_dict, model)

    load_info = model.load_state_dict(state_dict, strict=strict_load)
    if use_safetensors:
        validate_load_info(load_info)

    model.to(device).eval()
    return model


def get_rosie(checkpoint_dir: Path, cfg_model: OmegaConf, device: torch.device) -> torch.nn.Module:
    nc_out = len(cfg_model.data.targ_channel_names)
    model = get_rosie_model(num_outputs=nc_out)
    checkpoint_path = checkpoint_dir / "rosie.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def get_hemit(checkpoint_dir: Path, cfg_model: OmegaConf, device: torch.device, img_size: Tuple[int, int]) -> torch.nn.Module:
    nc_out = len(cfg_model.data.targ_channel_names)
    config = {
        "model_params": {
            "input_nc": 3,
            "output_nc": nc_out,
            "img_size": list(img_size),
            "patch_size": 4,
            "window_size": 8,
            "depths": [2, 2, 6, 2],
            "embed_dim": 96,
        }
    }
    model = ResnetGeneratorSwinT(**config["model_params"])
    state_dict = torch.load(checkpoint_dir / "best.pth", map_location="cpu")
    state_dict = adapt_checkpoint_hemit(state_dict, model)
    state_dict = resize_embed_hemit_statedict(state_dict, model)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def get_diffusion_ft(checkpoint_dir: Path, device: torch.device):
    """Load the diffusion fine-tuning pipeline."""
    torch.cuda.empty_cache()
    pipe = load_diffusion_pipeline(str(checkpoint_dir), device)
    return pipe
