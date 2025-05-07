
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

from cycle_gan_model import get_cyclegan_model

import sys
sys.path.append('../../')
from src.dataset import TileSlideDataset


BATCH_SIZE = ...
TORCH_WEIGHTS = ...


def cycle_gan_inference(dataframe,):

    

    dataset = TileSlideDataset(
            dataframe=dataframe,
            preprocess_input_fn=None)

    num_workers = os.cpu_count() - 1
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=False)


    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_cyclegan_model()
    model = model.eval().to(device)

    ### LOAD CHECKPOINT


    for batch in tqdm(dataloader, total=len(dataloader), leave=False):
        tile_names = batch["tile_name"]
        with torch.no_grad():
            mask = batch.float().std(axis=1, keepdims=True) < 3
            mask = mask * (batch.float().mean(axis=1, keepdims=True) > 220)
            out_batch = model(normalize(batch.float() / 255))
            out_batch = out_batch / 2 + 0.5
            out_batch = torch.clip(out_batch, 0., 1.)
            out_batch = (out_batch * 255).to(torch.uint8).cpu()
            out_batch = torch.where(mask, torch.ones_like(out_batch) * 250, out_batch)
            out_batch = torch.permute(out_batch, (0, 2, 3, 1)).numpy()
        
        cv2.imwrite(
            str(Path(out_dir) / f"{tile_name}.png"),
            out_batch[0].astype(np.uint8)
        )
            

    del tile, out_batch
