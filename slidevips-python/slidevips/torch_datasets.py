"""
PyTorch Dataset classes for efficient tile/patch extraction from WSIs using SlideVips.

Includes datasets for standard tile extraction and image-to-image translation tasks with paired
WSIs.
"""

import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .reader import SlideVips


if "LD_PRELOAD" not in os.environ:
    warnings.warn(
        "\n"
        + "=" * 80
        + "\n⚠️  WARNING: jemalloc is NOT preloaded.\n"
        + "⚠️  Using SlideVips torch datasets without jemalloc may cause uncontrolled RAM growth.\n"
        + "⚠️  To enable jemalloc, first install it:\n"
        + "⚠️      sudo apt-get install libjemalloc2\n"
        + "⚠️  Then run your script like this:\n"
        + "⚠️      LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python your_script.py\n"
        + "=" * 80 + "\n",
        RuntimeWarning,
        stacklevel=2
    )


class SlideDataset(Dataset):
    """
    A PyTorch Dataset for reading regions/tiles directly from whole slide images using SlideVips.

    This dataset manages the loading, augmentation, and preprocessing of image tiles extracted from
    large slide images. It supports spatial and color augmentations, custom preprocessing and
    filtering functions, and efficient slide file management.

    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing slide metadata (slide names, paths,
            etc.).
        dataframe (pd.DataFrame): DataFrame containing tile metadata (positions, slide names, etc.).
        channel_idxs (list or None): Indices of channels to extract from the slide images.
        mode (str): Image mode (e.g., "RGB" or "mIF").
        preprocess_input_fn (Callable or None): Function to preprocess input images
            (e.g. Normalization).
        filter_input_fn (Callable or None): Function to apply on input numpy images before
            augmentation and normalization.
        spatial_augmentations (Callable or None): Function for spatial augmentations.
        color_augmentations (Callable or None): Function for color augmentations.
        reiter_fetch (bool): Whether to re-iterate fetching tiles. Set to False.
        scale_factor (float or None): Optional scale factor for resizing slides.
            Not recommended, it is very slow. Use rather an interpolation function in
            spatial_augmentations.
    Attributes:
        df (pd.DataFrame): DataFrame containing tile metadata.
        slide_name2path (dict): Mapping from slide names to their file paths.
        slide_in_dict (dict): Cache of opened SlideVips objects for each slide.
        channel_idxs (list or None): Indices of channels to extract from the slide images.
        mode (str): Image mode (e.g., "RGB").
        preprocess_input_fn (Callable or None): Function to preprocess input images.
        filter_input_fn (Callable or None): Function to apply on input numpy images before
            augmentation and normalization.
        spatial_augmentations (Callable or None): Function for spatial augmentations.
        color_augmentations (Callable or None): Function for color augmentations.
        reiter_fetch (bool): Whether to re-iterate fetching tiles.
        scale_factor (float or None): Optional scale factor for resizing slides.
    Raises:
        AssertionError: If any tile in `dataframe` does not have a corresponding slide in
            `slide_dataframe`.
        ValueError: If `scale_factor` is not positive.
    Example:
        >>> dataset = SlideDataset(slide_dataframe, dataframe, mode="RGB")
        >>> sample = dataset[0]
        >>> image, tile_name = sample["image"], sample["tile_name"]
    """

    def __init__(self,
                 slide_dataframe: pd.DataFrame,
                 dataframe: pd.DataFrame,
                 channel_idxs: List[int] = None,
                 mode: str = "RGB",
                 preprocess_input_fn: Optional[Callable] = None,
                 filter_input_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 no_concurrency: bool = True,
                 reiter_fetch: bool = False,
                 scale_factor: Optional[float] = None,
                 ):
        """Initialize a SlideDataset instance."""
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

        self.no_concurrency = no_concurrency
        if scale_factor is not None:
            if scale_factor <= 0:
                raise ValueError("scale_factor should be positive")
            elif scale_factor == 1.:
                scale_factor = None
        self.scale_factor = scale_factor
        self.reiter_fetch = reiter_fetch

    @classmethod
    def from_one_slide(cls, slide_path, tile_positions, level, tile_size, channel_idxs=None,
                       mode="RGB", preprocess_input_fn=None, filter_input_fn=None,
                       spatial_augmentations=None, color_augmentations=None, reiter_fetch=False,
                       scale_factor=None):
        """
        Create a SlideDataset for a single slide using simpler initialization.

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

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Dict[torch.Tensor, str]: Dictionary containing the image tensor and tile name.
        """
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
            if self.no_concurrency:
                slide_in.set_concurrency(1)
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

        return {"image": image, "tile_name": tile_name, "slide_name": slide_name,
                "x": row["x"], "y": row["y"], "level": level,
                "tile_size_x": tile_size[0], "tile_size_y": tile_size[1]}

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def reset(self) -> None:
        """Close all open slides and clears the slide dictionary."""
        for slide in self.slide_in_dict.values():
            slide.close()
        self.slide_in_dict.clear()

    def __del__(self):
        """Destructor that calls the reset method to clean up resources."""
        self.reset()


class Img2ImgSlideDataset(Dataset):
    """
    PyTorch Dataset using WSI reader for image-to-image translation tasks using aligned WSIs.

    This dataset is an extension of `SlideDataset` that allows to read pairs of images directly
    from WSIs. It is designed for scenarios where both input and target WSIs are available and
    spatially aligned. It enables paired patch extraction for supervised learning.
    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing slide metadata (in_slide_name),
            including paths for input (input_slide_path) and target (target_slide_path) slides.
        dataframe (pd.DataFrame): DataFrame specifying patch coordinates, slide names, levels, and
            tile sizes.
        in_channel_idxs (Optional[List[int]], optional): Indices of channels to extract from input
            slides. Defaults to None (keeping all).
        targ_channel_idxs (Optional[List[int]], optional): Indices of channels to extract from
            target slides. Defaults to None (keeping all).
        mode_in (str, optional): Color mode for input slides (e.g., "RGB"). Defaults to "RGB".
        mode_targ (str, optional): Color mode for target slides (e.g., "RGB"). Defaults to "RGB".
        preprocess_input_fn (Optional[Callable], optional): Function to preprocess input images.
            Defaults to None.
        preprocess_target_fn (Optional[Callable], optional): Function to preprocess target images.
            Defaults to None.
        filter_target_fn (Optional[Callable], optional): Function to apply on target numpy images
            before augmentation and normalization. Defaults to None.
        spatial_augmentations (Optional[Callable], optional): Callable for spatial augmentations
            applied to both input and target (same transformation). Defaults to None.
        color_augmentations (Optional[Callable], optional): Callable for color augmentations
            applied to input images only. Defaults to None.
        reiter_fetch (bool, optional): Whether to reinitialize slide objects on each fetch.
            Defaults to False.
    Attributes:
        df (pd.DataFrame): DataFrame with patch information.
        inslide_name2path (dict): Mapping from input slide names to file paths.
        targslide_name2path (dict): Mapping from target slide names to file paths.
        slide_in_dict (dict): Cache of opened input slide objects.
        slide_targ_dict (dict): Cache of opened target slide objects.
        in_channel_idxs (list): Channel indices for input slides.
        targ_channel_idxs (list): Channel indices for target slides.
        mode_in (str): Color mode for input slides.
        mode_targ (str): Color mode for target slides.
        preprocess_input_fn (Callable): Preprocessing function for input images.
        preprocess_target_fn (Callable): Preprocessing function for target images.
        filter_target_fn (Callable): Filtering function for target images.
        spatial_augmentations (Callable): Spatial augmentation function.
        color_augmentations (Callable): Color augmentation function.
        reiter_fetch (bool): Whether to reinitialize slide objects on each fetch.
    Returns:
        dict: A dictionary with keys:
            - "image": torch.Tensor, input image patch (C, H, W)
            - "target": torch.Tensor, target image patch (C, H, W)
            - "tile_name": str, unique identifier for the patch
    Example:
        >>> dataset = Img2ImgSlideDataset(slide_dataframe, dataframe)
        >>> sample = dataset[0]
        >>> image, target = sample["image"], sample["target"]
    """

    def __init__(self,
                 slide_dataframe: pd.DataFrame,
                 dataframe: pd.DataFrame,
                 in_channel_idxs: List[int] = None,
                 targ_channel_idxs: List[int] = None,
                 mode_in: str = "RGB",
                 mode_targ: str = "RGB",
                 preprocess_input_fn: Optional[Callable] = None,
                 preprocess_target_fn: Optional[Callable] = None,
                 filter_target_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 no_concurrency: bool = True,
                 reiter_fetch: bool = False,
                 ):
        """Initialize a Img2ImgSlideDataset instance."""
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

        self.no_concurrency = no_concurrency
        self.reiter_fetch = reiter_fetch

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[torch.Tensor, torch.Tensor, str]: Dictionary containing the image and target
                tensors and tile name.
        """
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
            if self.no_concurrency:
                slide_in.set_concurrency(1)
            self.slide_in_dict[slide_name] = slide_in
        try:
            slide_targ = self.slide_targ_dict[slide_name]
        except KeyError:
            slide_targ = SlideVips(
                self.targslide_name2path[slide_name], self.targ_channel_idxs,
                self.mode_targ, self.reiter_fetch)
            if self.no_concurrency:
                slide_targ.set_concurrency(1)
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
        """Return the number of samples in the dataset."""
        return len(self.df)

    def reset(self) -> None:
        """Destructor that calls the reset method to clean up resources."""
        self.slide_in_dict.clear()
        self.slide_targ_dict.clear()

    def __del__(self):
        """Destructor that calls the reset method to clean up resources."""
        self.reset()
