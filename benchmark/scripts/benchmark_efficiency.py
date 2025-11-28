"""
Benchmark FLOPs, #parameters, and VRAM usage for H&E→mIF models.

Usage (from MIPHEI-ViT repo root):
    python scripts/benchmark_efficiency.py --device cuda --height 256 --width 256
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
from fvcore.nn import FlopCountAnalysis
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# =============================================================================
# Repo / imports
# =============================================================================

from benchmark.models import (
    get_miphei,
    get_hemit,
    get_pix2pix,
    get_rosie,
    get_diffusion_ft,
)
from benchmark.models.rosie import infer_sliding_window


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark H&E→mIF models: FLOPs, params, VRAM."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        help="Root directory containing model checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Input width.",
    )
    return parser.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def measure_vram_inference(forward_fn, device: torch.device) -> int:
    """
    Measure peak VRAM usage (in bytes) during forward_fn().

    Parameters
    ----------
    forward_fn : callable
        A function with no arguments that performs a forward pass
        (e.g., lambda: model(x)).
    device : torch.device
        CUDA device used for measurement.

    Returns
    -------
    peak_bytes : int
        Peak VRAM usage in bytes.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("VRAM measurement requires CUDA.")

    dev_str = device if isinstance(device, str) else device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_str)
    torch.cuda.synchronize(dev_str)

    with torch.no_grad():
        forward_fn()

    torch.cuda.synchronize(dev_str)
    peak_bytes = torch.cuda.max_memory_allocated(dev_str)
    return peak_bytes


def pretty_flops(n: float) -> str:
    """Turn raw FLOPs into readable units."""
    if n > 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n > 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n > 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    return f"{n:.3f} FLOPs"


def compute_flops_module(model: torch.nn.Module, *example_inputs) -> float:
    """Measure FLOPs from a nn.Module forward() call."""
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, example_inputs).total()
    return flops


def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Per-model benchmarking
# =============================================================================

def benchmark_pix2pix(checkpoints_dir: Path,
                      device: torch.device,
                      H: int,
                      W: int,
                      x: torch.Tensor) -> Dict[str, Any]:
    name = "Pix2Pix"
    ckpt_dir = checkpoints_dir / "pix2pix"
    print(f"\n=== {name} ===")
    cfg_model = OmegaConf.load(ckpt_dir / "config.yaml")
    model = get_pix2pix(ckpt_dir, cfg_model, device).to(device).eval()

    # VRAM
    vram_bytes = measure_vram_inference(lambda: model(x), device)
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    # FLOPs & params
    flops = compute_flops_module(model, x)
    n_params, _ = count_params(model)
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")
    print(f"[{name}] #parameters: {n_params:,}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params,
    }


def benchmark_miphei_vit(checkpoints_dir: Path,
                         device: torch.device,
                         H: int,
                         W: int,
                         x: torch.Tensor) -> Dict[str, Any]:
    name = "MIPHEI-ViT"
    ckpt_dir = checkpoints_dir / "MIPHEI-vit"
    print(f"\n=== {name} ===")
    cfg_model = OmegaConf.load(ckpt_dir / "config.yaml")
    model = get_miphei(ckpt_dir, cfg_model, device, H, W)

    vram_bytes = measure_vram_inference(lambda: model(x), device)
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    flops = compute_flops_module(model, x)
    n_params, _ = count_params(model)
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")
    print(f"[{name}] #parameters: {n_params:,}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params,
    }


def benchmark_miphei_convnext(checkpoints_dir: Path,
                              device: torch.device,
                              H: int,
                              W: int,
                              x: torch.Tensor) -> Dict[str, Any]:
    name = "MIPHEI-ConvNeXt"
    ckpt_dir = checkpoints_dir / "MIPHEI-convnext"
    print(f"\n=== {name} ===")

    cfg_model = OmegaConf.load(ckpt_dir / "config.yaml")
    model = get_miphei(ckpt_dir, cfg_model, device, H, W)

    vram_bytes = measure_vram_inference(lambda: model(x), device)
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    flops = compute_flops_module(model, x)
    n_params, _ = count_params(model)
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")
    print(f"[{name}] #parameters: {n_params:,}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params,
    }


def benchmark_rosie(checkpoints_dir: Path,
                    device: torch.device,
                    H: int,
                    W: int,
                    x: torch.Tensor) -> Dict[str, Any]:
    name = "ROSIE"
    ckpt_dir = checkpoints_dir / "rosie_orion"
    print(f"\n=== {name} ===")

    cfg_model = OmegaConf.load(ckpt_dir / "config.yaml")
    model = get_rosie(ckpt_dir, cfg_model, device)

    # VRAM: sliding-window inference with padding
    x_rosie_vram = torch.nn.functional.pad(x, (64, 64, 64, 64), mode="reflect")
    vram_bytes = measure_vram_inference(
        lambda: infer_sliding_window(x_rosie_vram, model, P=128, S=8),
        device,
    )
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    # FLOPs: one forward on 224x224 patch, then multiply by number of sliding windows
    x_rosie_flops = torch.randn(1, 3, 224, 224, device=device)
    flops_patch = compute_flops_module(model, x_rosie_flops)
    sliding_windows_mult = (H // 8) * (W // 8)
    flops = flops_patch * sliding_windows_mult

    n_params, _ = count_params(model)
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")
    print(f"[{name}] #parameters: {n_params:,}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params,
    }


def benchmark_hemit(checkpoints_dir: Path,
                    device: torch.device,
                    H: int,
                    W: int,
                    x: torch.Tensor) -> Dict[str, Any]:
    name = "HEMIT"
    ckpt_dir = checkpoints_dir / "HEMIT"
    print(f"\n=== {name} ===")

    cfg_model = OmegaConf.load(ckpt_dir / "config.yaml")
    model = get_hemit(ckpt_dir, cfg_model, device, img_size=(H, W))

    vram_bytes = measure_vram_inference(lambda: model(x), device)
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    flops = compute_flops_module(model, x)
    n_params, _ = count_params(model)
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")
    print(f"[{name}] #parameters: {n_params:,}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params,
    }


def benchmark_diffusion_ft(checkpoints_dir: Path,
                           device: torch.device,
                           H: int,
                           W: int,
                           x: torch.Tensor) -> Dict[str, Any]:
    name = "Diffusion-FT"
    ckpt_dir = checkpoints_dir / "diffusion_ft"
    print(f"\n=== {name} ===")

    pipe = get_diffusion_ft(ckpt_dir, device)

    # VRAM: full pipeline call on x
    vram_bytes = measure_vram_inference(lambda: pipe(x), device)
    vram_mib = vram_bytes / 2**20
    print(f"[{name}] Peak VRAM: {vram_mib:.2f} MiB")

    nc_out = 16

    with torch.no_grad():
        vae_out = pipe.vae.encoder(x)
        vae_out_q = pipe.vae.quant_conv(vae_out)
        rgb_latent = torch.chunk(vae_out_q, chunks=2, dim=1)[0]
        noise_latent = torch.zeros_like(rgb_latent)
        x_latent = torch.cat([rgb_latent, noise_latent], dim=1)
        enc = pipe.empty_encoding
        T = pipe.scheduler.config.num_train_timesteps
        t = torch.full((1,), T - 1, device=device, dtype=torch.long)
        marker_embeds = pipe.marker_embeds[0]  # (M, D)
        marker_embeds = marker_embeds.unsqueeze(0)

    flops_vae_encoder = compute_flops_module(pipe.vae.encoder, x)
    flops_vae_encoder += compute_flops_module(pipe.vae.quant_conv, vae_out)
    flops_unet = FlopCountAnalysis(
        pipe.unet, (x_latent, t, enc, marker_embeds)
    ).total()
    flops_vae_decoder = compute_flops_module(pipe.vae.post_quant_conv, noise_latent)
    flops_vae_decoder += compute_flops_module(pipe.vae.decoder, noise_latent)

    flops = flops_vae_encoder + nc_out * flops_unet + nc_out * flops_vae_decoder
    print(f"[{name}] FLOPs per image: {pretty_flops(flops)}")

    # params: pipeline
    #n_params_pipe, _ = count_params(pipe)
    n_params_vae, _ = count_params(pipe.vae)
    n_params_unet, _ = count_params(pipe.unet)
    n_params_pipe = n_params_vae + n_params_unet

    print(f"[{name}] #parameters (full pipeline): {n_params_pipe:,}")

    del pipe
    torch.cuda.empty_cache()

    return {
        "name": name,
        "vram_mib": vram_mib,
        "flops": flops,
        "flops_str": pretty_flops(flops),
        "params": n_params_pipe,  # or n_params_pipe if you prefer
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Model':20s} {'VRAM (MiB)':>12s} {'FLOPs':>18s} {'#params':>15s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:20s} "
            f"{r['vram_mib']:12.2f} "
            f"{r['flops_str']:>18s} "
            f"{r['params']:15,d}"
        )
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    H, W = args.height, args.width
    checkpoints_dir = Path(args.checkpoints_dir)
    print(f"Using checkpoints_dir = {checkpoints_dir}")
    print(f"Input size: 1x3x{H}x{W}")
    print(f"Device: {device}")

    # single example input for all "direct" models
    x = torch.randn(1, 3, H, W, device=device)

    results = []

    # Order: you can comment out models you don't want.
    results.append(benchmark_pix2pix(checkpoints_dir, device, H, W, x))
    results.append(benchmark_miphei_vit(checkpoints_dir, device, H, W, x))
    results.append(benchmark_miphei_convnext(checkpoints_dir, device, H, W, x))
    results.append(benchmark_rosie(checkpoints_dir, device, H, W, x))
    results.append(benchmark_hemit(checkpoints_dir, device, H, W, x))
    results.append(benchmark_diffusion_ft(checkpoints_dir, device, H, W, x))

    print_summary(results)


if __name__ == "__main__":
    main()
