#!/usr/bin/env python
"""
Generate inference figure for multiple H&E→mIF models on a chosen tile.

Usage:
    python scripts/generate_figure_predictions.py \
        --slide-index 10197 \
        --checkpoint_dir /root/workdir/MIPHEI-ViT/checkpoints \
        ----output_dir figures/paper_fig2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from functools import partial
import pyvips
from PIL import Image
from omegaconf import OmegaConf

import tifffile
import albumentations as A

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# --- existing repo imports -------------------------------------------
from benchmark.models.rosie import retrieve_image_scale

from benchmark.models import (
    get_miphei,
    get_hemit,
    get_pix2pix,
    get_rosie,
    get_diffusion_ft,
)
# ---------------------------------------------------------------------

# ============================================================
# Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_dir", type=str,
                        help="Root directory containing model checkpoints.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save prediction PNGs.")
    parser.add_argument("--slide-index", type=int, default=10197,
                        help="Row index in dataframe to extract tile from.")
    parser.add_argument("--data-config", type=str, default="configs/data/orion.yaml",
                        help="Path to dataset config YAML.")

    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================

def normalize_tile(tile: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize 256×256 H&E tile."""
    x = torch.from_numpy(tile)
    x = (x - mean) / std
    x = x.permute((2, 0, 1)).unsqueeze(0).float()
    return x


def postprocess_output(out: torch.Tensor, a: float, b: float) -> np.ndarray:
    """
    Convert model output from [-a, b] to [0, 255].
    """
    arr = out.squeeze(0).permute((1, 2, 0))
    arr = (arr + a) / b
    arr = torch.clamp(arr, 0, 1) * 255
    return arr.to(torch.uint8).cpu().numpy()


def create_rgb_visual(mif_tile: np.ndarray) -> np.ndarray:
    """
    Convert 16-channel mIF tile to RGB visualization
    using channels [-2, 10, 0], normalized at 99.9th percentile.
    """
    rgb = mif_tile[..., [-2, 10, 0]].copy()
    rgb = rgb / np.percentile(rgb, 99.9, axis=(0, 1), keepdims=True)
    rgb = np.clip(rgb, 0, 1) * 255
    return rgb.astype(np.uint8)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_root = Path(args.checkpoints_dir)

    # --------------------------------------------------------
    # Load dataset row
    # --------------------------------------------------------
    cfg_data = OmegaConf.load(args.data_config)
    df = pd.read_csv(cfg_data.data.test_dataframe_path)
    row = df.iloc[args.slide_index]

    # --------------------------------------------------------
    # Load H&E and ground truth mIF tile
    # --------------------------------------------------------
    x1, y1, x2, y2 = [38, 38, 294, 294] # central crop for 333x333 of OrionCRC
    tile_he = np.asarray(Image.open(row["image_path"]).crop((x1, y1, x2, y2)))

    # Read target channels (skip channel 13 to match notebook)
    target_img = pyvips.Image.new_from_file(row["target_path"])
    channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16]
    target_mif = target_img[channels].crop(x1, y1, 256, 256).numpy()

    # Save H&E tile
    Image.fromarray(tile_he).save(output_dir / "tile_he.png")

    # ============================================================
    # HEMIT
    # ============================================================
    ckpt = checkpoints_root / "HEMIT"
    cfg_model = OmegaConf.load(ckpt / "config.yaml")
    hemit = get_hemit(ckpt, cfg_model, device, img_size=[256, 256])

    mean = torch.tensor(cfg_model.data.normalization.mean).view(1, 1, -1)
    std  = torch.tensor(cfg_model.data.normalization.std).view(1, 1, -1)

    x_norm = normalize_tile(tile_he, mean, std).to(device)

    with torch.inference_mode():
        out_hemit = hemit(x_norm)

    out_hemit = postprocess_output(out_hemit, a=1.0, b=2.0)
    out_vis_hemit = create_rgb_visual(out_hemit)
    Image.fromarray(out_vis_hemit).save(output_dir / "pred_hemit.png")

    del hemit; torch.cuda.empty_cache()

    # ============================================================
    # MIPHEI-ViT
    # ============================================================
    ckpt = checkpoints_root / "MIPHEI-vit"
    cfg_model = OmegaConf.load(ckpt / "config.yaml")
    miphei = get_miphei(ckpt, cfg_model, device, H=256, W=256).half()

    mean = torch.tensor(cfg_model.data.normalization.mean).view(1, 1, -1)
    std  = torch.tensor(cfg_model.data.normalization.std).view(1, 1, -1)

    x_norm = normalize_tile(tile_he, mean, std).to(device).half()

    with torch.inference_mode():
        out = miphei(x_norm).float()

    out_miphei = postprocess_output(out, a=0.9, b=1.8)
    out_vis_miphei = create_rgb_visual(out_miphei)
    Image.fromarray(out_vis_miphei).save(output_dir / "pred_miphei.png")

    del miphei; torch.cuda.empty_cache()

    # ============================================================
    # ROSIE
    # ============================================================
    ckpt = checkpoints_root / "rosie_orion"
    cfg_model = OmegaConf.load(ckpt / "config.yaml")
    rosie = get_rosie(ckpt, cfg_model, device)

    rosie_name = Path(row["image_path"]).stem + ".tiff"
    rosie_path = ckpt / "pred_orion" / rosie_name

    pred_rosie = tifffile.imread(rosie_path)[:32, :32]

    resize_fn = A.Lambda(partial(retrieve_image_scale, shape_crop=(256, 256)))
    out_rosie = resize_fn(image=pred_rosie)["image"]

    out_vis_rosie = create_rgb_visual(out_rosie)
    Image.fromarray(out_vis_rosie).save(output_dir / "pred_rosie.png")

    del rosie; torch.cuda.empty_cache()

    # ============================================================
    # Pix2Pix
    # ============================================================
    ckpt = checkpoints_root / "pix2pix"
    cfg_model = OmegaConf.load(ckpt / "config.yaml")
    pix2pix = get_pix2pix(ckpt, cfg_model, device)

    mean = torch.tensor(cfg_model.data.normalization.mean).view(1, 1, -1).float()
    std  = torch.tensor(cfg_model.data.normalization.std).view(1, 1, -1).float()

    x_norm = normalize_tile(tile_he, mean, std).to(device)

    with torch.inference_mode():
        out = pix2pix(x_norm)

    out_pix = postprocess_output(out, a=1.0, b=2.0)
    out_vis_pix = create_rgb_visual(out_pix)
    Image.fromarray(out_vis_pix).save(output_dir / "pred_pix2pix.png")

    del pix2pix; torch.cuda.empty_cache()

    # ============================================================
    # Diffusion FT
    # ============================================================
    ckpt = checkpoints_root / "diffusion_ft"
    diffusion_ft = get_diffusion_ft(ckpt, device)

    cfg_model = OmegaConf.load(ckpt / "config.yaml")
    mean = torch.tensor(cfg_model.data.normalization.mean).view(1, 1, -1).float()
    std  = torch.tensor(cfg_model.data.normalization.std).view(1, 1, -1).float()

    x_norm = normalize_tile(tile_he, mean, std).to(device)

    with torch.inference_mode():
        out = diffusion_ft(x_norm)[0]

    out_diff = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    out_vis_diff = create_rgb_visual(out_diff)
    Image.fromarray(out_vis_diff).save(output_dir / "pred_diffusion_ft.png")

    del diffusion_ft; torch.cuda.empty_cache()

    # ============================================================
    # Save ground truth
    # ============================================================
    out_vis_target = create_rgb_visual(target_mif)
    Image.fromarray(out_vis_target).save(output_dir / "target.png")

    print(f"\nAll predictions saved in: {output_dir}\n")


if __name__ == "__main__":
    main()
