from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .reader import SlideVips


class SlideDataset(Dataset): # also from slide directly

    def __init__(self,
                 slide_dataframe: pd.DataFrame,
                 dataframe: pd.DataFrame,
                 channel_idxs=None,
                 mode="RGB",
                 preprocess_input_fn: Optional[Callable] = None,
                 filter_input_fn: Optional[Callable] = None,
                 spatial_augmentations = None,
                 color_augmentations = None,
                 reiter_fetch=False,
                 scale_factor=None,
                 ):
        """Works only on registered examples"""
        #  slide dataframe and dataframe or only one ?
        assert dataframe["in_slide_name"].isin(slide_dataframe["in_slide_name"].tolist()).all()
        slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(
            dataframe["in_slide_name"].unique())]

        self.df = dataframe
        self.slide_name2path = slide_dataframe.set_index("in_slide_name")["in_slide_path"].to_dict()

        self.slide_in_dict = {}
        self.channel_idxs = channel_idxs
        self.mode = mode

        self.preprocess_input_fn = preprocess_input_fn
        self.filter_input_fn = filter_input_fn

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations

        self.reiter_fetch = reiter_fetch
        if scale_factor is not None:
            if scale_factor <= 0:
                raise ValueError("scale_factor should be positive")
            elif scale_factor == 1.:
                scale_factor = None
        self.scale_factor = scale_factor

    @classmethod
    def from_one_slide(cls, slide_path, tile_positions, level, tile_size, channel_idxs=None,
                 mode="RGB", preprocess_input_fn=None, filter_input_fn=None,
                 spatial_augmentations=None, color_augmentations=None, reiter_fetch=False,
                 scale_factor=None):
        
        """
        Create a SlideDataset from a single slide.
        
        Args:
            slide_path (str): Path to the slide.
            tile_positions (ndarray): Array of tile positions.
            tile_size (tuple): Size of the tiles.
            level (int): Level of the slide.
        """
        slide_name = Path(slide_path).stem
        slide_dataframe = pd.DataFrame({"in_slide_name": [slide_name],
                                        "in_slide_path": [slide_path]})
        dataframe = pd.DataFrame(columns=["in_slide_name", "x", "y", "level",
                                          "tile_size_x", "tile_size_y"])
        dataframe["x"] = tile_positions[..., 0]
        dataframe["y"] = tile_positions[..., 1]
        dataframe["in_slide_name"] = slide_name
        dataframe["level"] = level
        dataframe["tile_size_x"] = tile_size[0]
        dataframe["tile_size_y"] = tile_size[1]
        return cls(slide_dataframe, dataframe, channel_idxs, mode,
                   preprocess_input_fn, filter_input_fn, spatial_augmentations,
                   color_augmentations, reiter_fetch, scale_factor)

    def reset(self):
        for slide in self.slide_in_dict.values():
            slide.close()
        self.slide_in_dict.clear()

    def __getitem__(self, idx):
        # load images and target
        row = self.df.iloc[idx]
        slide_name = row["in_slide_name"]
        location = (row["x"], row["y"])
        level = row["level"]
        tile_size = (row["tile_size_x"], row["tile_size_y"])
        tile_name = "_".join(map(str, [slide_name, *location, level, *tile_size]))

        try:
            slide_in = self.slide_in_dict[slide_name]
        except KeyError:
            slide_in = SlideVips(self.slide_name2path[slide_name],
                                 self.channel_idxs, self.mode, self.reiter_fetch)
            if self.scale_factor is not None:
                slide_in.resize(self.scale_factor)
            self.slide_in_dict[slide_name] = slide_in

        image = slide_in.read_region(location, level, tile_size)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.dtype not in [np.uint8, np.float32]:
            image = np.float32(image)

        if self.filter_input_fn:
            image = self.filter_input_fn(image)

        if self.spatial_augmentations:
            image = self.spatial_augmentations(image=image)["image"]

        if self.color_augmentations:
            image = self.color_augmentations(image=image)["image"]
            image = np.clip(image, 0, 255)

        if self.preprocess_input_fn:
            image = self.preprocess_input_fn(image)

        image = torch.from_numpy(image).permute(2, 0, 1)

        return {"image": image, "tile_name": tile_name}

    def __len__(self):
        return len(self.df)

    def __del__(self):
        self.reset()


class Img2ImgSlideDataset(Dataset): # also from slide directly

    def __init__(self,
                 slide_dataframe: pd.DataFrame,
                 dataframe: pd.DataFrame,
                 in_channel_idxs=None,
                 targ_channel_idxs=None,
                 mode_in="RGB",
                 mode_targ="RGB",
                 preprocess_input_fn: Optional[Callable] = None,
                 preprocess_target_fn: Optional[Callable] = None,
                 filter_target_fn: Optional[Callable] = None,
                 spatial_augmentations = None,
                 color_augmentations = None,
                 reiter_fetch=False,
                 ):
        """Works only on registered examples"""
        #  slide dataframe and dataframe or only one ?
        assert dataframe["in_slide_name"].isin(slide_dataframe["in_slide_name"].tolist()).all()
        slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(
            dataframe["in_slide_name"].unique())]

        self.df = dataframe
        self.inslide_name2path = slide_dataframe.set_index(
            "in_slide_name")["in_slide_path"].to_dict()
        self.targslide_name2path = slide_dataframe.set_index(
            "in_slide_name")["targ_slide_path"].to_dict()

        self.slide_in_dict = {}
        self.slide_targ_dict = {}
        self.in_channel_idxs = in_channel_idxs
        self.targ_channel_idxs = targ_channel_idxs

        self.mode_in = mode_in
        self.mode_targ = mode_targ

        self.preprocess_input_fn = preprocess_input_fn
        self.preprocess_target_fn = preprocess_target_fn

        self.filter_target_fn = filter_target_fn
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations

        self.reiter_fetch = reiter_fetch

    def reset(self):
        self.slide_in_dict.clear()
        self.slide_targ_dict.clear()

    def __getitem__(self, idx):
        # load images and target
        row = self.df.iloc[idx]
        slide_name = row["in_slide_name"]
        location = (row["x"], row["y"])
        level = row["level"]
        tile_size = (row["tile_size_x"], row["tile_size_y"])
        tile_name = "_".join(map(str, [slide_name, *location, level, *tile_size]))

        try:
            slide_in = self.slide_in_dict[slide_name]
        except KeyError:
            slide_in = SlideVips(
                self.inslide_name2path[slide_name], self.in_channel_idxs,
                self.mode_in, self.reiter_fetch)
            self.slide_in_dict[slide_name] = slide_in
        try:
            slide_targ = self.slide_targ_dict[slide_name]
        except KeyError:
            slide_targ = SlideVips(
                self.targslide_name2path[slide_name], self.targ_channel_idxs,
                self.mode_targ, self.reiter_fetch)
            self.slide_targ_dict[slide_name] = slide_targ

        image = slide_in.read_region(location, level, tile_size)
        target = slide_targ.read_region(location, level, tile_size)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(target.shape) == 2:
            target = np.expand_dims(target, axis=-1)

        if image.dtype not in [np.uint8, np.float32]:
            image = np.float32(image)
        if target.dtype not in [np.uint8, np.float32]:
            target = np.float32(target)

        if self.filter_target_fn:
            target = self.filter_target_fn(target)

        if self.spatial_augmentations:
            transformed = self.spatial_augmentations(image=image, image_target=target)
            image, target = transformed["image"], transformed["image_target"]

        if self.color_augmentations:
            image = self.color_augmentations(image=image)["image"]
            image = np.clip(image, 0, 255)

        if self.preprocess_input_fn:
            image = self.preprocess_input_fn(image)
        if self.preprocess_target_fn:
            target = self.preprocess_target_fn(target)

        image = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)

        return {"image": image, "target": target, "tile_name": tile_name}

    def __len__(self):
        return len(self.df)
