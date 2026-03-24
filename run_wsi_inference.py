"""
WSI H&E to mIF inference script.

This script performs WSI prediction using a trained checkpoint.
Given a WSI H&E file, it tiles the image, runs inference on each tile, and assembles the
predictions into a multi-channel OME-TIFF output file containing the prediction results.

Main features:
- Loads a trained model checkpoint for inference.
- Tiles the input WSI, processes each tile, and reconstructs the prediction.
- Outputs an OME-TIFF file with prediction channels, preserving spatial metadata.

Coordinate system
-----------------
Three pixel spaces are used throughout:
  model  — tile_size × tile_size, the resolution the model processes (at mpp_target)
  slide  — pixel coordinates at the chosen pyramid level
  lvl0   — pixel coordinates at level 0 (required by SlideDataset / get_locs_otsu)

Conversions:
  scale            = mpp_target / slide_mpp
  tile_size_slide  = round(tile_size * scale)          [model → slide]
  downsample       = slide.level_downsamples[level]
  x_slide          = round(x_lvl0 / downsample)        [lvl0 → slide]
  tile_size_lvl0   = round(tile_size_slide * downsample)  [slide → lvl0]
"""

import argparse
import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from safetensors.torch import load_file

from slidevips import SlideVips
from slidevips.ome_metadata import adapt_ome_metadata
from slidevips.tiling import get_locs_otsu
from slidevips.torch_datasets import SlideDataset

from src.dataset import NormalizationLayer
from src.generators import get_generator
from src.utils import validate_load_info


def _flush_row(row_buffer, row_y_key, temp_root, n_channels, slide_w_level, tile_stride_slide):
    """Write per-channel row numpy buffer to disk and release reference."""
    for ch in range(n_channels):
        ch_data = np.ascontiguousarray(row_buffer[:, :, ch])
        img = pyvips.Image.new_from_memory(
            ch_data.tobytes(), slide_w_level, tile_stride_slide, 1, "uchar"
        )
        img.write_to_file(os.path.join(temp_root, f"row_{row_y_key}_c{ch}.v"))


def wsi_inference(slide_path, checkpoint_dir, output_dir, num_workers=-1, level=0, tile_size=256,
                  tile_overlap=10, batch_size=4, mpp_target=0.5) -> None:
    """
    Perform WSI inference using a trained model and save the output as an OME-TIFF file.

    This function processes an H&E WSI by tiling it, running inference on each tile using
    a trained model, and reconstructing the output into a pyramidal multi-channel OME-TIFF
    image. Inference is performed row by row: for each row of tiles, predictions are stored
    on disk as per-channel vips cache files. After all rows are processed, the full WSI is
    assembled by joining rows per channel, then stacking channels vertically (required for
    QuPath OME-TIFF compatibility).

    Args:
        slide_path (str or Path): Path to the input H&E WSI file.
        checkpoint_dir (str or Path): Directory containing the model checkpoint
            ('model.weights.ckpt') and configuration file ('config.yaml').
        output_dir (str or Path): Directory where the output OME-TIFF will be saved.
        level (int, optional): Pyramid level of the input WSI to process. Defaults to 0.
        tile_size (int, optional): Model input size in pixels (at mpp_target). Defaults to 256.
        tile_overlap (int, optional): Overlap border (in model pixels) on each side of each
            tile. Discarded after inference to produce seamless output. Defaults to 10.
        batch_size (int, optional): Number of tiles per inference batch. Defaults to 4.
        mpp_target (float, optional): Target microns-per-pixel for model inference.
            Defaults to 0.5.
    """
    # ------------------------------------------------------------------
    # Paths and output setup
    # ------------------------------------------------------------------
    slide_name = Path(slide_path).stem
    Path(output_dir).mkdir(exist_ok=True)
    output_path = str(Path(output_dir) / f"{slide_name}.ome.tiff")

    # ------------------------------------------------------------------
    # Slide metadata
    # ------------------------------------------------------------------
    slide = SlideVips(slide_path)

    slide_mpp = slide.level_resolutions[level]           # MPP at chosen level
    downsample = np.mean(slide.level_downsamples[level]) # level-0 → chosen level scale

    slide_dim_lvl0 = slide.dimensions                    # (W, H) at level 0
    slide_w_level, slide_h_level = slide.level_dimensions[level]     # slide-level shape for row canvas

    thumbnail = slide.get_thumbnail((3000, 3000))
    thumbnail = np.ones(thumbnail.shape[:2], dtype=bool)

    # ------------------------------------------------------------------
    # Coordinate calculations
    # ------------------------------------------------------------------
    scale = mpp_target / slide_mpp
    if np.isclose(scale, 1.0):
        scale = 1.0

    # Tile geometry in slide-level pixels
    tile_size_slide   = int(round(tile_size    * scale))
    tile_overlap_slide = int(round(tile_overlap * scale))
    tile_stride_slide  = tile_size_slide - 2 * tile_overlap_slide  # content size (no border)

    if tile_stride_slide <= 0:
        raise ValueError(
            f"tile_stride_slide={tile_stride_slide} <= 0. "
            "Reduce tile_overlap or increase tile_size."
        )

    # Tile geometry in level-0 pixels (for get_locs_otsu / SlideDataset)
    # get_locs_otsu stride = tile_size_lvl0 - tile_overlap_lvl0 = tile_stride_slide * downsample
    # => tile_overlap_lvl0 = 2 * tile_overlap_slide * downsample
    tile_size_lvl0    = int(round(tile_size_slide        * downsample))
    tile_overlap_lvl0 = int(round(2 * tile_overlap_slide * downsample))

    # ------------------------------------------------------------------
    # Tiling: compute tile positions on level-0 grid
    # ------------------------------------------------------------------
    tile_positions, _ = get_locs_otsu(
        thumbnail, slide_dim_lvl0, tile_size_lvl0, tile_overlap=tile_overlap_lvl0
    )
    # Sort row-major (Y then X) so row transitions can be detected in order
    tile_positions = tile_positions[np.lexsort((tile_positions[:, 0], tile_positions[:, 1]))]

    # DataFrames for SlideDataset (positions are level-0 coordinates)
    slide_dataframe = pd.DataFrame({
        "in_slide_path": [slide_path],
        "in_slide_name": [slide_name],
    })
    dataframe = pd.DataFrame({
        "in_slide_name": slide_name,
        "x":             tile_positions[:, 0],
        "y":             tile_positions[:, 1],
        "level":         level,
        "tile_size_x":   tile_size_slide,
        "tile_size_y":   tile_size_slide,
    })

    slide.close()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(str(Path(checkpoint_dir) / "config.yaml"))
    channel_names = cfg.data.targ_channel_names
    n_channels = len(channel_names)

    # Instantiate generator model
    generator = get_generator(cfg.model.model_name, tile_size, 3, n_channels, cfg)

    # Load model weights from checkpoint
    checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
    state_dict = load_file(checkpoint_path, device="cpu")
    load_info = generator.load_state_dict(state_dict, strict=False)
    validate_load_info(load_info)

    # Set model to evaluation mode and half precision
    generator = generator.eval().to(device)

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    input_mean = torch.Tensor(cfg.data.normalization.mean).view((1, -1, 1, 1)).to(device)
    input_std = torch.Tensor(cfg.data.normalization.std).view((1, -1, 1, 1)).to(device)

    dataset = SlideDataset(
        slide_dataframe=slide_dataframe,
        dataframe=dataframe,
        preprocess_input_fn=None,
        reiter_fetch=False,
    )
    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    # ------------------------------------------------------------------
    # Inference — row-by-row tile assembly
    # ------------------------------------------------------------------
    temp_root = os.path.join(tempfile.gettempdir(), slide_name)
    os.makedirs(temp_root, exist_ok=True)

    current_row_y_lvl0 = None   # level-0 Y key of the in-progress row
    # numpy row buffer: (row_height, slide_width, n_channels) — filled by numpy slicing,
    # avoiding a long lazy pyvips insert() chain which is slow for wide slides
    row_buffer = None
    row_y_positions_lvl0 = []   # ordered list of flushed row Y keys (level-0)

    for batch in tqdm(dataloader):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                image = batch["image"].to(device)
                image = (image.float() - input_mean) / input_std

                # Resize slide-level tile to model input size if MPP differs
                if scale != 1.0:
                    # scale > 1: slide tile is bigger → downsample to model size → area
                    # scale < 1: slide tile is smaller → upsample to model size → bicubic
                    mode_in = "area" if scale > 1.0 else "bicubic"
                    kw_in = {"align_corners": False} if mode_in == "bicubic" else {}
                    image = F.interpolate(
                        image, size=(tile_size, tile_size), mode=mode_in, **kw_in
                    )

                out_batch = generator(image).float()

                # Remove overlap in model space FIRST, then resize to slide scale.
                # Cropping before resize avoids interpolation ringing at the tile boundary
                # that would bleed into the content region when resizing the full tile.
                tile_stride_model = tile_size - 2 * tile_overlap
                out_batch = out_batch[
                    ...,
                    tile_overlap:tile_overlap + tile_stride_model,
                    tile_overlap:tile_overlap + tile_stride_model,
                ]

                # Resize content region back to slide-level stride (if MPP differs)
                if scale != 1.0:
                    # scale > 1: upsample back → bicubic
                    # scale < 1: downsample back → area
                    mode_out = "bicubic" if scale > 1.0 else "area"
                    kw_out = {"align_corners": False} if mode_out == "bicubic" else {}
                    out_batch = F.interpolate(
                        out_batch,
                        size=(tile_stride_slide, tile_stride_slide),
                        mode=mode_out, **kw_out,
                    )
                # out_batch shape: (batch, C, tile_stride_slide, tile_stride_slide)

        # Normalise predictions to uint8
        out_batch = (out_batch.clamp(-0.9, 0.9) + 0.9) / 1.8 * 255
        out_batch = out_batch.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        # out shape per tile: (tile_stride_slide, tile_stride_slide, n_channels) — content only

        # Tile positions are level-0 coordinates (returned directly by SlideDataset)
        xs_lvl0 = batch["x"].numpy()
        ys_lvl0 = batch["y"].numpy()

        for x_lvl0, y_lvl0, out in zip(xs_lvl0, ys_lvl0, out_batch):
            # Convert level-0 position to slide-level position
            x_slide = int(round(x_lvl0 / downsample))

            # Detect row transition (Y key changes)
            if current_row_y_lvl0 is None or y_lvl0 != current_row_y_lvl0:
                if row_buffer is not None:
                    _flush_row(row_buffer, current_row_y_lvl0, temp_root,
                               n_channels, slide_w_level, tile_stride_slide)
                    row_y_positions_lvl0.append(current_row_y_lvl0)

                current_row_y_lvl0 = y_lvl0
                row_buffer = np.zeros(
                    (tile_stride_slide, slide_w_level, n_channels), dtype=np.uint8
                )

            # out is already content-only; insert at slide-level x position (after left overlap)
            insert_x = x_slide + tile_overlap_slide
            end_x = min(insert_x + tile_stride_slide, slide_w_level)
            w = end_x - insert_x
            if w > 0:
                row_buffer[:, insert_x:end_x, :] = out[:, :w, :]

    # Flush the final row
    if row_buffer is not None:
        _flush_row(row_buffer, current_row_y_lvl0, temp_root,
                   n_channels, slide_w_level, tile_stride_slide)
        row_y_positions_lvl0.append(current_row_y_lvl0)

    # ------------------------------------------------------------------
    # Final assembly: stack rows per channel, then stack channels
    # ------------------------------------------------------------------
    ys_sorted_lvl0 = sorted(row_y_positions_lvl0)
    row_height = tile_stride_slide  # content height per row (slide-level pixels)

    channel_mosaics = []

    for ch in range(n_channels):
        ch_rows = []
        """if tile_overlap_slide > 0:
            blank = pyvips.Image.black(slide_w_level, tile_overlap_slide, bands=1).cast("uchar")
            ch_rows.append(blank)"""
        prev_end_slide = tile_overlap_slide  # slide-level Y end of the previous row

        for y_lvl0 in ys_sorted_lvl0:
            y_slide = int(round(y_lvl0 / downsample))
            row_start_slide = y_slide + tile_overlap_slide  # content start (after top overlap)

            # Insert blank rows to fill any gap between non-contiguous tissue bands
            gap = row_start_slide - prev_end_slide
            if gap > 0:
                print(gap / row_height)
                n_blanks = int(round(gap / row_height))
                blank = pyvips.Image.black(slide_w_level, row_height, bands=1).cast("uchar")
                ch_rows.extend([blank] * n_blanks)

            fn = os.path.join(temp_root, f"row_{y_lvl0}_c{ch}.v")
            ch_rows.append(pyvips.Image.new_from_file(fn, access="sequential"))
            prev_end_slide = row_start_slide + row_height
        mosaic_ch = pyvips.Image.arrayjoin(ch_rows, across=1)
        slide_w, slide_h = min(slide_w_level, mosaic_ch.width), min(slide_h_level, mosaic_ch.height)
        mosaic_ch = mosaic_ch.crop(0, 0, slide_w, slide_h)
        channel_mosaics.append(pyvips.Image.arrayjoin(mosaic_ch, across=1))

    # ------------------------------------------------------------------
    # Output: save as pyramidal OME-TIFF
    # ------------------------------------------------------------------
    # Stack channels vertically (required for QuPath OME-TIFF format)
    stacked = pyvips.Image.arrayjoin(channel_mosaics, across=1).copy(interpretation="b-w")

    magnification = int(10 / slide_mpp)
    ome_xml_metadata = adapt_ome_metadata(stacked, slide_mpp, channel_names, magnification)

    image_height = stacked.height // n_channels
    stacked.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    stacked.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    pbar_filesave = tqdm(total=100, unit="Percent", desc="Writing Output WSI", position=0,
                         leave=True)

    def eval_cb(image, progress):
        pbar_filesave.update(progress.percent - pbar_filesave.n)

    stacked.set_progress(True)
    stacked.signal_connect("eval", eval_cb)

    stacked.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / slide_mpp,
        yres=1000 / slide_mpp,
        page_height=image_height,
    )
    del stacked
    shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_path",      type=str,   required=True,  help="Path to input WSI file")
    parser.add_argument("--checkpoint_dir",  type=str,   required=True,  help="Checkpoint directory")
    parser.add_argument("--output_dir",      type=str,   required=True,  help="Output directory")

    parser.add_argument("--num_workers",     type=int,   default=-1,    help="Number of workers in dataloader")
    parser.add_argument("--level",           type=int,   default=0,      help="Pyramid level")
    parser.add_argument("--tile_size",       type=int,   default=512,    help="Model tile size (px)")
    parser.add_argument("--tile_overlap",    type=int,   default=20,     help="Overlap border (model px)")
    parser.add_argument("--batch_size",      type=int,   default=8,     help="Inference batch size")
    parser.add_argument("--mpp_target",      type=float, default=0.5,    help="Target MPP for inference")
    args = parser.parse_args()

    start_time = time.time()
    wsi_inference(
        args.slide_path, args.checkpoint_dir, args.output_dir,
        num_workers=args.num_workers,
        level=args.level, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        batch_size=args.batch_size, mpp_target=args.mpp_target,
    )
    elapsed = time.time() - start_time
    print(f"Inference done for {Path(args.slide_path).stem} in {timedelta(seconds=elapsed)}.")
