import pyvips

import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A


class ORIONDataset(Dataset):
    """
    Paired H&E → mIF tiles for image-to-image (diffusion) training.
    
    Returns:
        {
            "rgb_norm": Tensor [3,H,W],
            "target"  : Tensor [C,H,W],   # C = 1 or 3
            "marker_id": int,
            "tile_name": str,
            "slide_name": str (optional),
        }
    """

    def __init__(
        self, data_dir: str, split: str
    ):
        if split == "train":
            df_path = str(Path(data_dir) / "train_dataframe.csv")
        elif split == "val":
            df_path = str(Path(data_dir) / "val_dataframe.csv")
        elif split == "vis":
            df_path = str(Path(data_dir) / "val_dataframe.csv")
        else:
            raise ValueError(f"Unknown split {split} for ORIONDataset")
        self.df = pd.read_csv(df_path)

        if split == "val":
            self.df = self.df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        elif split == "vis":
            self.df = self.df.sample(frac=0.01, random_state=42).reset_index(drop=True)

        self.targ_channel_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
        self.spatial_augmentations = A.Compose([
            A.RandomCrop(width=256, height=256),
        ], additional_targets={'image_target': 'image'})
        self.repeat_target_to_3ch = False
        self.disp_name = "orion"
        self.filename_ls_path = None
        self.num_channels = len(self.targ_channel_idxs)
        self.augmentation_dir = Path(data_dir) / "he_augmented"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        input_path  = row["image_path"]
        if np.random.uniform() < 0.1:
            aug_path = self.augmentation_dir / Path(input_path).name
            input_path = str(aug_path)
        target_path = row["target_path"]

        tile_name = Path(input_path).stem

        # --- load H&E / RGB ---
        image = np.asarray(Image.open(input_path))   # (H,W,C)

        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        
        # choose a marker
        marker_id = np.random.randint(len(self.targ_channel_idxs))
        marker_ch = self.targ_channel_idxs[marker_id]

        # --- load mIF ---
        mif_target = pyvips.Image.new_from_file(
            target_path)
        mif_target = mif_target[marker_ch].numpy()

        if mif_target.ndim == 2:
            mif_target = np.expand_dims(mif_target, -1)


        # repeat to 3ch only if needed for VAE
        if self.repeat_target_to_3ch:
            mif_target = np.repeat(mif_target, 3, axis=-1)  # (H,W,3)

        # --- types ---
        image = image.astype(np.float32)
        mif_target = mif_target.astype(np.float32)

        # --- augments ---
        if self.spatial_augmentations:
            transformed = self.spatial_augmentations(image=image, image_target=mif_target)
            image       = transformed["image"]
            mif_target  = transformed["image_target"]


        # --- preprocess ---
        image_norm = (image.astype(np.float32) / 127.5) - 1.0
        mif_target = (mif_target.astype(np.float32) / 127.5) - 1.0

        # --- to torch ---
        image_norm = torch.from_numpy(image_norm).permute(2,0,1)        # [C,H,W]
        mif_target = torch.from_numpy(mif_target).permute(2,0,1)

        out = {
            "rgb": image_norm,
            "target":   mif_target,
            "marker_id": marker_id,
            "rgb_relative_path": tile_name,
        }

        return out
