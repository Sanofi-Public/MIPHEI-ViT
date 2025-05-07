
import pyvips
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
import numpy as np

import os
from tqdm import tqdm

from slidevips import SlideVips
from slidevips.torch_datasets import SlideDataset
from slidevips.tiling import get_locs_otsu
from pathlib import Path
import functools
import matplotlib.pyplot as plt

from slidevips.ome_metadata import adapt_ome_metadata
from slidevips.tiling import order_tiles_horizontally

from cycle_gan_model import ResnetGenerator

RESOLUTION_LVL_0 = ...
LEVEL = ...
TILE_SIZE = ...
TILE_OVERLAP = ...
BATCH_SIZE = ...
TORCH_WEIGHTS = ...


def cycle_gan_inference(slide_path,):

    slide_name = Path(slide_path).stem

    slide_dataframe = pd.DataFrame(columns=["in_slide_name", "in_slide_path"])
    slide_dataframe["in_slide_name"] = [slide_name]
    slide_dataframe["in_slide_path"] = slide_path


    slide_he = SlideVips(slide_path)
    mpp = slide_he.mpp
    scale_factor = RESOLUTION_LVL_0 / mpp

    thumbnail = slide_he.get_thumbnail((3000, 3000))
    tile_size_lvl0 = np.int32(np.round(tile_shift * scale_factor / np.mean(slide_he.level_downsamples[LEVEL])))
    tile_positions, _ = get_locs_otsu(
        thumbnail, slide_he.dimensions, tile_size_lvl0, TILE_OVERLAP, 0.01)
    slide_he.close()
    idxs = order_tiles_horizontally(tile_positions)
    tile_positions = tile_positions[idxs]

    dataframe = pd.DataFrame(columns=["in_slide_name", "x", "y", "level", "tile_size_x", "tile_size_y"])
    dataframe["in_slide_name"] = [slide_name] * len(tile_positions)
    dataframe["x"] = tile_positions[..., 0]
    dataframe["y"] = tile_positions[..., 1]
    dataframe["level"] = LEVEL
    dataframe["tile_size_x"] = int(TILE_SIZE * scale_factor) ###
    dataframe["tile_size_y"] = int(TILE_SIZE * scale_factor)

    dataset = SlideDataset(
            slide_dataframe=slide_dataframe,
            dataframe=dataframe,
            mode="RGB",
            preprocess_input_fn=None,
            spatial_augmentations=A.Resize(TILE_SIZE, TILE_SIZE, always_apply=True), ####
            reiter_fetch=False) #####################

    num_workers = 0 #####################
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=False)


    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    model = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
    model = model.eval().to(device)

    ### LOAD CHECKPOINT


    final_img = pyvips.Image.black(slide_dim[0], slide_dim[1], bands=1).cast(pyvips.BandFormat.UCHAR) #### White background

    for idx_batch, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            mask = batch.float().std(axis=1, keepdims=True) < 3
            mask = mask * (batch.float().mean(axis=1, keepdims=True) > 220)
            out_batch = model(normalize(batch.float() / 255))
            out_batch = out_batch / 2 + 0.5
            out_batch = torch.clip(out_batch, 0., 1.)
            out_batch = (out_batch * 255).to(torch.uint8).cpu()
            out_batch = torch.where(mask, torch.ones_like(out_batch) * 250, out_batch)
            out_batch = torch.permute(out_batch, (0, 2, 3, 1)).numpy()
        
        tile_positions_batch = tile_positions[idx_batch * BATCH_SIZE: (idx_batch + 1) * BATCH_SIZE]
        for tile_position, out in zip(tile_positions_batch, out_batch):
            tile_position_scale = np.int32(np.round(tile_position / scale_factor)) + TILE_OVERLAP
            tile = pyvips.Image.new_from_array(out)
            tile = tile.crop(TILE_OVERLAP, TILE_OVERLAP, tile_shift, tile_shift)
            final_img = final_img.insert(tile,
                tile_position_scale[0] + tile.height, 
                tile_position_scale[1] + tile.width)
            

    del tile, out_batch
