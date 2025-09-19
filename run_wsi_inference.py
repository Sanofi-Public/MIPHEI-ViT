"""
WSI H&E to mIF inference script.

This script performs WSI prediction using a trained checkpoint.
Given a WSI H&E file, it tiles the image, runs inference on each tile, and assembles the predictions
into a multi-channel OME-TIFF output file containing the prediction results.
Main features:
- Loads a trained model checkpoint for inference.
- Tiles the input WSI, processes each tile, and reconstructs the prediction.
- Outputs an OME-TIFF file with prediction channels, preserving spatial metadata.
"""

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from slidevips import SlideVips
from slidevips.ome_metadata import adapt_ome_metadata
from slidevips.tiling import get_locs_otsu
from slidevips.torch_datasets import SlideDataset

from src.dataset import NormalizationLayer
from src.generators import get_generator
from src.utils import validate_load_info


def wsi_inference(slide_path, checkpoint_dir, output_dir, level=0, tile_size=2048, tile_overlap=10,
                  batch_size=4, mpp_target=0.5) -> None:
    """
    Perform WSI inference using a trained model and save the output as an OME-TIFF file.

    This function processes an H&E WSI by tiling it, running inference on each tile using a trained
    model, and reconstructing the output into a pyramidal multi-channel OME-TIFF image. It handles
    tiling, normalization, batching, model loading, and final image assembly.

    Inference is performed row by row across the WSI: for each row of tiles, predictions are run and
    the resulting large row images (per channel) are stored temporarily on disk. This approach is
    efficient and avoids RAM issues with large slides. After all rows are processed, each channel's
    full WSI image is reconstructed by vertically stacking the corresponding row images from disk.
    Finally, all channel images are stacked vertically to match the OME-TIFF format required for
    QuPath compatibility. Storing independent channel images on disk makes the stacking operation
    much faster and more memory efficient. The output is saved as an OME-TIFF file, and the
    temporary folder is removed at the end. Output can be read in QuPath.
    Args:
        slide_path (str or Path): Path to the input H&E WSI file.
        checkpoint_dir (str or Path): Directory containing the model checkpoint and configuration
            files.
        output_dir (str or Path): Directory where the output OME-TIFF will be saved.
        level (int, optional): Pyramid level of the input H&E WSI to process.
            Defaults to 0 (highest resolution).
        tile_size (int, optional): Size of the tiles (in pixels) to extract from the slide.
            Defaults to 2048.
        tile_overlap (int, optional): Overlap (in pixels) between adjacent tiles. Defaults to 10.
        batch_size (int, optional): Number of tiles to process in a batch during inference.
            Defaults to 4.
        mpp_target (float, optional): Target microns-per-pixel (MPP) resolution for processing.
            Defaults to 0.5.
    Returns:
        None
    Side Effects:
        - Writes the output OME-TIFF file to `output_dir`.
        - Creates and deletes temporary files/directories during processing.
    Raises:
        FileNotFoundError: If input files or required configuration files are missing.
        RuntimeError: If model loading or inference fails.
        ValueError: If input parameters are invalid.
    Notes:
        - Requires GPU for optimal performance.
        - Inference done in half precision.
        - Uses pyvips for efficient image processing and assembly.
    """
    # ---------------------------------------------------------------
    #  Image Preprocessings: Extract medatada and tiling.
    # ---------------------------------------------------------------
    # Get the slide name (without extension) for output file naming
    slide_name = Path(slide_path).stem
    output_path = str(Path(output_dir) / f"{slide_name}.ome.tiff")

    # Create a DataFrame to hold slide metadata
    slide_dataframe = pd.DataFrame({
        "in_slide_path": [slide_path]})
    slide_dataframe["in_slide_name"] = slide_name

    # Open the slide using SlideVips to extract metadata
    slide = SlideVips(slide_path)
    slide_mpp = slide.level_resolutions[level]  # microns-per-pixel at selected level

    slide_dim = slide.dimensions  # (width, height) at level 0
    thumbnail = slide.get_thumbnail((3000, 3000))  # for tissue detection/tiling

    # Compute scale factor to match target MPP
    scale = mpp_target / slide_mpp
    if np.isclose(scale, 1.0):
        scale = 1.0  # avoid tiny floating point errors

    # Calculate tile size and overlap in slide pixels (at selected level)
    tile_size_slide = int(np.round(tile_size * scale))
    tile_overlap_slide = int(np.round(tile_overlap * scale))
    tile_shift_slide = tile_size_slide - 2 * tile_overlap_slide  # stride between tiles

    # Compute tiling grid on level-0
    tile_size_lvl0 = int(np.round(tile_shift_slide /
                                  np.mean(slide.level_downsamples[level])))
    tile_overlap_lvl0 = int(np.round(tile_overlap_slide /
                                     np.mean(slide.level_downsamples[level])))

    # Get tile positions using Otsu tissue detection on the thumbnail
    tile_positions, _ = get_locs_otsu(
        thumbnail, slide_dim, tile_size_lvl0, tile_overlap=tile_overlap_lvl0
    )
    # Sort tile positions by Y then X to group by row
    tile_positions = tile_positions[np.lexsort((tile_positions[:, 0], tile_positions[:, 1]))]

    # Create a tile-level DataFrame describing each tile's location and properties
    dataframe = pd.DataFrame(
        columns=["in_slide_name", "x", "y", "level", "tile_size_x", "tile_size_y"]
    )
    dataframe["x"] = tile_positions[..., 0]
    dataframe["y"] = tile_positions[..., 1]
    dataframe["level"] = level
    dataframe["tile_size_x"] = int(tile_size_slide)
    dataframe["tile_size_y"] = int(tile_size_slide)
    dataframe["in_slide_name"] = slide_dataframe["in_slide_name"].iloc[0]

    slide.close()  # close SlideVips object

    # ---------------------------------------------------------------
    #  Model Loading.
    # ---------------------------------------------------------------
    # Free GPU memory before model loading
    torch.cuda.empty_cache()

    # Set device for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model configuration
    config_path = str(Path(checkpoint_dir) / "config.yaml")
    cfg = OmegaConf.load(config_path)

    # Model input and output channel setup
    nc_in = 3
    channel_names = cfg.data.targ_channel_names
    n_channels = len(channel_names)

    # Instantiate generator model
    generator = get_generator(cfg.model.model_name, tile_size, nc_in, n_channels, cfg)

    # Load model weights from checkpoint
    checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
    state_dict = load_file(checkpoint_path, device="cpu")
    load_info = generator.load_state_dict(state_dict, strict=False)
    validate_load_info(load_info)

    # Set model to evaluation mode and half precision
    generator = generator.eval().to(device).half()

    # ---------------------------------------------------------------
    #  Dataloader Loading.
    # ---------------------------------------------------------------
    # Create normalization layer for H&E input
    channel_stats_rgb = {"mean": cfg.data.normalization.mean,
                         "std": cfg.data.normalization.std}
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    # If scale is not 1.0, resize tiles to target size; otherwise, no spatial augmentation
    spatial_augmentations = A.Resize(height=tile_size, width=tile_size) if scale != 1.0 else None

    # Create WSI dataset and dataloader
    dataset = SlideDataset(
        slide_dataframe=slide_dataframe,
        dataframe=dataframe,
        preprocess_input_fn=preprocess_input_fn,
        spatial_augmentations=spatial_augmentations,
        reiter_fetch=False)

    num_workers = 0  # Use single worker for lower RAM usage (slower)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False)

    # ---------------------------------------------------------------
    #  Inference.
    # ---------------------------------------------------------------
    # Temporary directory for storing intermediate row images
    temp_root = os.path.join(tempfile.gettempdir(), slide_name)
    os.makedirs(temp_root, exist_ok=True)

    # Variables for row assembly
    current_row_y = None
    row_imgs = None
    row_y_positions = []

    # Iterate over batches of tiles from the dataloader
    for batch in tqdm(dataloader):
        with torch.inference_mode():
            # Run inference on the batch using the generator model
            out_batch = generator(batch["image"].to(device).half())
            out_batch = out_batch.float()
            # If the scale is not 1.0, resize predictions to match the slide resolution
            if scale != 1.0:
                out_batch = F.interpolate(
                    out_batch,
                    size=(tile_size_slide, tile_size_slide),
                    mode='bicubic' if scale > 1.0 else 'area',
                    align_corners=False
                )
            # Normalize and convert predictions to uint8 for image assembly
            out_batch = (out_batch.clamp(-0.9, 0.9) + 0.9) / 1.8 * 255
            out_batch = torch.permute(out_batch, (0, 2, 3, 1)).cpu().numpy().astype(np.uint8)

        # Extract tile (x, y) positions from tile names in the batch
        tile_positions_batch = np.asarray([tile_name.split("_")[-5:-3]
                                           for tile_name in batch["tile_name"]]).astype(np.int32)

        # Process each tile in the batch
        for (x, y), out in zip(tile_positions_batch, out_batch):
            # Create a pyvips image from the output array and crop to remove overlap
            tile = pyvips.Image.new_from_array(out).crop(tile_overlap_slide, tile_overlap_slide,
                                                         tile_shift_slide, tile_shift_slide)

            # If this is the first tile of a new row, allocate new row images for each channel
            if current_row_y is None or y != current_row_y:

                # If there is a previous row, flush (save) all channel images to disk
                if row_imgs is not None:
                    for ch in range(n_channels):
                        row_imgs[ch].write_to_file(
                            os.path.join(temp_root, f"row_{current_row_y}_c{ch}.v")
                        )
                        row_imgs[ch] = None
                    row_y_positions.append(current_row_y)
                    gc.collect()

                # Start a new row: create blank images for each channel
                current_row_y = y
                row_imgs = [
                    pyvips.Image.black(slide_dim[0], tile.height, bands=1
                                       ).cast("uchar")
                    for _ in range(n_channels)
                ]  # one image per channel for efficiency during arrayjoin in final assembly

            # Insert the current tile's band (channel) into the appropriate row image
            for ch in range(n_channels):
                band = tile.extract_band(ch)  # Get the channel band (no copy)
                row_imgs[ch] = row_imgs[ch].insert(
                    band,
                    x + tile_overlap_slide,
                    0
                )

    # After all batches, flush the last set of row images to disk
    if row_imgs is not None:
        for ch in range(n_channels):
            row_imgs[ch].write_to_file(
                os.path.join(temp_root, f"row_{current_row_y}_c{ch}.v")
            )
        row_y_positions.append(current_row_y)
        row_imgs = None
        gc.collect()

    # ---------------------------------------------------------------
    #  Final Assembly: Assemble the full-size channel mosaics from row images.
    # ---------------------------------------------------------------
    row_height = tile_shift_slide  # Height of each row (without overlap)
    final_slide_dim = slide.level_dimensions[level]  # (width, height) at selected level
    slide_w, slide_h = final_slide_dim
    ys_sorted = sorted(row_y_positions)  # Sorted list of Y positions for each row

    channel_mosaics = []  # List to hold the final mosaic for each channel

    for ch in range(n_channels):
        # Gather all row images for this channel in the correct order
        ch_rows = []
        prev_end = None  # Track the end Y position of the previous row

        for y in ys_sorted:
            # If there is a gap between rows, fill with blank rows to preserve spatial alignment
            if prev_end is not None:
                gap = y - prev_end
                if gap > 0:
                    n_blanks = int(round(gap / row_height)) - 1
                    if n_blanks > 0:
                        # Insert the required number of blank rows
                        blank = pyvips.Image.black(
                            slide_w, row_height, bands=1
                        ).cast("uchar")
                        ch_rows.extend([blank] * n_blanks)

            # Load the actual row image for this channel and Y position
            fn = os.path.join(temp_root, f"row_{y}_c{ch}.v")
            ch_rows.append(pyvips.Image.new_from_file(fn, access="sequential"))
            prev_end = y + row_height  # Update the end position for the next iteration

        # Vertically join all rows for this channel and crop to the slide height
        mosaic_ch = pyvips.Image.arrayjoin(ch_rows, across=1) \
                                .crop(0, 0, slide_w, slide_h)
        channel_mosaics.append(mosaic_ch)

    # ---------------------------------------------------------------
    #  Output Saving.
    # ---------------------------------------------------------------

    # Horizontally join all channel mosaics into a single image (bands are stacked vertically)
    # This step is mandatory to be readable in QuPath for OME.TIFF
    # We stored images per channel because it is much more efficient during this step
    stacked = pyvips.Image.arrayjoin(channel_mosaics, across=1
                                     ).copy(interpretation="b-w")

    # Calculate slide magnification for OME metadata (e.g., 20x, 40x)
    magnification = int(10 / slide_mpp)

    # Generate OME-XML metadata for the output file
    ome_xml_metadata = adapt_ome_metadata(stacked, slide_mpp, channel_names, magnification)

    # Each channel is stacked vertically, so compute the height of a single channel image
    # This step is necessary for OME-TIFF metadata. QuPath can then reconstruct the image correctly.
    image_height = stacked.height // n_channels

    # Set OME-TIFF metadata: page height and OME-XML description
    stacked.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    stacked.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    # Progress bar for file saving
    pbar_filesave = tqdm(total=100, unit="Percent", desc="Writing Output WSI", position=0,
                         leave=True)

    def eval_cb(image, progress):  # Callback to update progress bar during TIFF writing
        pbar_filesave.update(progress.percent - pbar_filesave.n)

    stacked.set_progress(True)
    stacked.signal_connect('eval', eval_cb)

    # Save the stacked image as a pyramidal, tiled, OME-TIFF file
    stacked.tiffsave(output_path,
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
                     page_height=image_height)

    # Clean up resources and delete temp folder
    del stacked
    shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=str, help='Slide Path')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint Directory')
    parser.add_argument('--output_dir', type=str, help='Output Directory')

    parser.add_argument('--level', type=int, default=0, help='Tile Level')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile Size')
    parser.add_argument('--tile_overlap', type=int, default=10, help='Tile Overlap')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    args = parser.parse_args()

    slide_path = args.slide_path
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    level = args.level
    tile_size = args.tile_size
    tile_overlap = args.tile_overlap
    batch_size = args.batch_size

    start_time = time.time()
    wsi_inference(slide_path, checkpoint_dir, output_dir, level=level,
                  tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size)
    elapsed = time.time() - start_time
    slide_name = Path(slide_path).stem
    print(f"Inference done for {slide_name} in {timedelta(seconds=elapsed)}.")
