"""Dataset classes and functions used for H&E to mIF translation."""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List
from omegaconf import DictConfig

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import pyvips
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from slidevips import SlideVips

from .augmentations import HedColorAugmentor


def get_augmentations(width: int, height: int, return_nuclei: bool = False,
                      training: bool = True) -> tuple:
    """
    Create spatial and color augmentations for image-to-image translation tasks.

    Spatial augmentations are applied to input, target and nuclei images (same transformation for
    associated images), while color augmentations are applied only to the input image.
    Augmentations differ between training and evaluation modes.
    Args:
        width (int): The width of the output image after cropping.
        height (int): The height of the output image after cropping.
        return_nuclei (bool, optional): If True, expects an additional nuclei mask from dataset.
            Defaults to False.
        training (bool, optional): If True, applies random spatial and color augmentations for
            training. If False, applies only center cropping for evaluation. Defaults to True.
    Returns:
        tuple:
            spatial_augmentations (albumentations.Compose): Composed spatial augmentations to be
                applied on both input and target images.
            color_augmentations (albumentations.Compose or None): Composed color augmentations to
                be applied only on the input image. None if not in training mode.
    """
    additional_targets = {'image_target': 'image'}
    if return_nuclei:
        additional_targets["nuclei"] = "image"
    if training:
        spatial_augmentations = A.Compose([
            A.RandomCrop(width=width, height=height),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(p=0.1, num_holes_range=[1, 1], hole_height_range=[0., 0.3],
                            hole_width_range=[0., 0.3])
        ], additional_targets=additional_targets)

        color_augmentations = A.Compose([
            HedColorAugmentor(thresh=0.015, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 1.5), p=0.1),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.1)
        ])
    else:
        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width, height=height),
        ], additional_targets=additional_targets)

        color_augmentations = None

    return spatial_augmentations, color_augmentations


def dataloader_worker_init_fn(worker_id: int) -> None:
    """
    Initialize each worker process for the DataLoader.

    This function is mainly used for WSIs datasets from SlideVIPS to address partial RAM issues by
    resetting the dataset state in each worker process.
    Args:
        worker_id (int): The worker process ID.

    Notes:
        This function is intended to be passed as the `worker_init_fn` argument to PyTorch's
            DataLoader. It retrieves the dataset instance for the current worker and calls its
            `reset()` method to ensure proper memory management.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


def get_width_height(dataframe: pd.DataFrame) -> Tuple[int, int]:
    """
    Get the width and height of a tile from a DataFrame.

    If the DataFrame does not contain an 'image_path' column, we are using a WSI reader and the
    width and height are taken from the 'tile_size_x' and 'tile_size_y' columns, respectively.
    Otherwise, if using a tile reader, the function opens the tile at the path specified in the
    'image_path' column and retrieves its dimensions.
    Args:
        dataframe (pd.DataFrame): The tile-level dataFrame containing image or tile information.
    Returns:
        tuple: A tuple (width, height) representing the dimensions of the image or tile.
    """
    from_slide = "image_path" not in dataframe.columns
    if from_slide:
        width = dataframe["tile_size_x"].iloc[0]
        height = dataframe["tile_size_y"].iloc[0]
    else:
        width, height = Image.open(dataframe["image_path"].iloc[0]).size

    return width, height


def get_effective_width_height(width: int, height: int, train: bool = False) -> Tuple[int, int]:
    """
    Calculate the effective width and height, optionally rounding down to the nearest power of 2 \
    during training.

    This is done to match UNet input size requirements and to ensure pipeline compatibility with
    other input shapes. In spatial augmentation, images will be cropped at this dimension.

    Args:
        width (int): The original width value.
        height (int): The original height value.
        train (bool, optional): If True, rounds width and height down to the nearest power of 2.
            Defaults to False.
    Returns:
        Tuple[int, int]: The effective width and height values.
    """
    if train:
        # Calculate the largest power of 2 less than or equal to width and height
        width = int(2 ** (np.floor(np.log2(width))))
        height = int(2 ** (np.floor(np.log2(height))))

    return width, height


class DataModule:
    """
    DataModule for managing datasets and dataloaders for training, validation, and testing.

    This class handles the creation and configuration of datasets and dataloaders for image-to-image
    tasks, supporting both slide-based and tile-based data sources. It provides flexible options for
    data augmentation, preprocessing, and sampling.
    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing slide-level metadata.
        train_dataframe (pd.DataFrame): DataFrame containing training tile-level data information.
        val_dataframe (pd.DataFrame): DataFrame containing validation tile-level data information.
        test_dataframe (pd.DataFrame): DataFrame containing test tile-level data information.
        targ_channel_idxs (list): List of target channel indices for the output images.
        batch_size (int): Batch size for dataloaders.
        input_shape (Tuple[int, int]): Input image shape as (width, height).
        from_slide (bool, optional): Whether to use slide-based datasets. Defaults to False.
            WARNING: Can cause RAM issue if True.
        pin_memory (bool, optional): Whether to use pinned memory in dataloaders. Defaults to True.
        return_nuclei (bool, optional): Whether to return nuclei masks. Defaults to False.
        train_sampler (Sampler, optional): Custom sampler for the training dataloader.
            Defaults to None.
        preprocess_input_fn (callable, optional): Function to preprocess input images,
            typically normalization. Defaults to None.
        preprocess_target_fn (callable, optional): Function to preprocess target images.
            Defaults to None.
    Attributes:
        slide_dataframe (pd.DataFrame): Slide-level metadata.
        train_dataframe (pd.DataFrame): Training tile-level data.
        val_dataframe (pd.DataFrame): Validation tile-level data.
        test_dataframe (pd.DataFrame): Test tile-level data.
        targ_channel_idxs (list): Target channel indices.
        batch_size (int): Batch size.
        from_slide (bool): Slide-based dataset flag.
        pin_memory (bool): Pin memory flag.
        return_nuclei (bool): Return nuclei flag.
        train_sampler (Sampler): Training sampler.
        preprocess_input_fn (callable): Input preprocessing function.
        preprocess_target_fn (callable): Target preprocessing function.
        input_shape (Tuple[int, int]): Input image shape.
        num_workers (int): Number of worker processes for dataloaders.
        train_dataset: Training dataset instance.
        val_dataset: Validation dataset instance.
        test_dataset: Test dataset instance.
        train_dataloader: Training dataloader instance.
        val_dataloader: Validation dataloader instance.
        test_dataloader: Test dataloader instance.
    Methods:
        setup():
            Initializes datasets and dataloaders for training, validation, and testing.
        get_dataloaders():
            Returns the train, validation, and test dataloaders.
    """

    def __init__(self, slide_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame,
                 val_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame,
                 targ_channel_idxs: list, batch_size: int, input_shape: Tuple[int, int],
                 from_slide: bool = False, pin_memory: bool = True,
                 return_nuclei: bool = False, train_sampler: Sampler = None,
                 preprocess_input_fn=None, preprocess_target_fn=None):
        self.slide_dataframe = slide_dataframe
        self.train_dataframe = train_dataframe
        self.val_dataframe = val_dataframe
        self.test_dataframe = test_dataframe

        self.targ_channel_idxs = targ_channel_idxs
        self.batch_size = batch_size
        self.from_slide = from_slide
        self.pin_memory = pin_memory
        self.return_nuclei = return_nuclei
        self.train_sampler = train_sampler
        self.preprocess_input_fn = preprocess_input_fn
        self.preprocess_target_fn = preprocess_target_fn
        self.input_shape = input_shape

        self.num_workers = os.cpu_count() - 1

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def setup(self):
        """Create datasets for train, validation, and test."""
        width, height = self.input_shape
        spatial_augmentations, color_augmentations = get_augmentations(
            width, height, return_nuclei=self.return_nuclei, training=True)
        test_spatial_transformation, _ = get_augmentations(
            width, height, return_nuclei=self.return_nuclei, training=False)
        if self.from_slide:
            self.train_dataset = Img2ImgNucleiSlideDataset(
                slide_dataframe=self.slide_dataframe,
                dataframe=self.train_dataframe,
                targ_channel_idxs=self.targ_channel_idxs,
                mode_in="RGB",
                mode_targ="IF",
                preprocess_input_fn=self.preprocess_input_fn,
                preprocess_target_fn=self.preprocess_target_fn,
                spatial_augmentations=spatial_augmentations,
                color_augmentations=color_augmentations,
                return_nuclei=self.return_nuclei,
                reiter_fetch=True)
            self.val_dataset = Img2ImgNucleiSlideDataset(
                slide_dataframe=self.slide_dataframe,
                dataframe=self.val_dataframe,
                targ_channel_idxs=self.targ_channel_idxs,
                mode_in="RGB",
                mode_targ="IF",
                preprocess_input_fn=self.preprocess_input_fn,
                preprocess_target_fn=self.preprocess_target_fn,
                return_nuclei=self.return_nuclei,
                reiter_fetch=True)
            self.test_dataset = Img2ImgNucleiSlideDataset(
                    slide_dataframe=self.slide_dataframe,
                    dataframe=self.test_dataframe,
                    targ_channel_idxs=self.targ_channel_idxs,
                    mode_in="RGB",
                    mode_targ="IF",
                    preprocess_input_fn=self.preprocess_input_fn,
                    preprocess_target_fn=self.preprocess_target_fn,
                    return_nuclei=self.return_nuclei,
                    reiter_fetch=True)
        else:  # from tiles
            self.train_dataset = TileImg2ImgSlideDataset(
                self.train_dataframe,
                targ_channel_idxs=self.targ_channel_idxs,
                preprocess_input_fn=self.preprocess_input_fn,
                preprocess_target_fn=self.preprocess_target_fn,
                spatial_augmentations=spatial_augmentations,
                color_augmentations=color_augmentations,
                return_nuclei=self.return_nuclei)
            self.val_dataset = TileImg2ImgSlideDataset(
                self.val_dataframe,
                targ_channel_idxs=self.targ_channel_idxs,
                preprocess_input_fn=self.preprocess_input_fn,
                preprocess_target_fn=self.preprocess_target_fn,
                spatial_augmentations=test_spatial_transformation,
                return_nuclei=self.return_nuclei)
            self.test_dataset = TileImg2ImgSlideDataset(
                    self.test_dataframe,
                    targ_channel_idxs=self.targ_channel_idxs,
                    preprocess_input_fn=self.preprocess_input_fn,
                    preprocess_target_fn=self.preprocess_target_fn,
                    spatial_augmentations=test_spatial_transformation,
                    return_nuclei=self.return_nuclei)

        shuffle_train = self.train_sampler is None
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            sampler=self.train_sampler, shuffle=shuffle_train,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=dataloader_worker_init_fn)
        self.val_dataset.reset()
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory,
            worker_init_fn=dataloader_worker_init_fn)
        self.test_dataset.reset()
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory,
            worker_init_fn=dataloader_worker_init_fn)

    def get_dataloaders(self):
        """Return the train, validation, and test dataloaders."""
        return self.train_dataloader, self.val_dataloader, self.test_dataloader


class TileSlideDataset(Dataset):
    """
    A PyTorch Dataset for loading image tiles and optional nuclei masks for inference.

    This dataset is designed for inference tasks where each sample consists of an image tile,
    optionally with a nuclei mask. It supports channel selection, spatial and color
    augmentations, and custom preprocessing functions.
    Args:
        dataframe (pd.DataFrame): Tile-level dataFrame with at least an "image_path" column,
            and optionally "nuclei_path".
        channel_idxs (list or None, optional): Indices of channels to select from the image.
            Defaults to None (all).
        preprocess_input_fn (Callable, optional): Function to preprocess the input image.
            Defaults to None.
        spatial_augmentations (callable, optional): Function to apply spatial augmentations.
            Defaults to None.
        color_augmentations (callable, optional): Function to apply color augmentations.
            Defaults to None.
        return_nuclei (bool, optional): If True, loads and returns nuclei masks. Defaults to False.
    Attributes:
        df (pd.DataFrame): Tile-level dataFrame containing image paths and optional nuclei paths.
        channel_idxs (list or None): Indices of channels to select from the input image.
        preprocess_input_fn (Callable, optional): Function to preprocess the input image.
        spatial_augmentations (callable, optional): Function to apply spatial augmentations.
        color_augmentations (callable, optional): Function to apply color augmentations.
        return_nuclei (bool): Whether to return nuclei masks along with images.
    Example:
        dataset = TileSlideDataset(
            dataframe=df,
            channel_idxs=None,
            preprocess_input_fn=preprocess_fn,
            spatial_augmentations=spatial_aug,
            color_augmentations=color_aug,
            return_nuclei=True
        )
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 channel_idxs: List[int] = None,
                 preprocess_input_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 return_nuclei: bool = False,
                 ):
        self.df = dataframe

        self.channel_idxs = channel_idxs

        self.preprocess_input_fn = preprocess_input_fn

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.return_nuclei = return_nuclei

    def __getitem__(self, idx: int) -> dict:
        """Load an image tile at idx and optionally its nuclei mask."""
        row = self.df.iloc[idx]
        input_path = row["image_path"]
        tile_name = Path(input_path).stem

        image = np.asarray(Image.open(input_path))

        output_dict = {}
        if self.return_nuclei:
            nuclei_path = row["nuclei_path"]
            nuclei = pyvips.Image.new_from_file(
                nuclei_path, page=0, access="sequential").numpy()
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if self.channel_idxs is not None:
            image = image[..., self.channel_idxs]

        if image.dtype not in [np.uint8, np.float32]:
            image = np.float32(image)

        if self.spatial_augmentations:
            if self.return_nuclei:
                transformed = self.spatial_augmentations(
                    image=image, nuclei=np.int32(nuclei))
                image = transformed["image"]
                nuclei = np.uint32(transformed["nuclei"])
            else:
                transformed = self.spatial_augmentations(image=image)
                image = transformed["image"]

        if self.color_augmentations:
            image = self.color_augmentations(image=image)["image"]
            image = np.clip(image, 0, 255)

        if self.preprocess_input_fn:
            image = self.preprocess_input_fn(image)

        if not image.flags.writeable:
            image = image.copy()
        image = torch.from_numpy(image).permute(2, 0, 1)

        output_dict.update({"image": image, "tile_name": tile_name})
        if self.return_nuclei:
            nuclei = torch.from_numpy(nuclei)
            output_dict.update(
                {"nuclei": nuclei})

        if "in_slide_name" in row.keys():
            output_dict["slide_name"] = row["in_slide_name"]
        return output_dict

    def reset(self):
        """Only for compatibility with WSIs datasets."""
        pass

    def __len__(self):
        return len(self.df)


class TileImg2ImgSlideDataset(Dataset):
    """
    PyTorch Dataset for loading paired image and target tiles from a DataFrame, with optional \
    nuclei masks and augmentations.

    This dataset is designed for image-to-image tasks (e.g., image translation, segmentation) where
    each sample consists of an input image tile, a corresponding target tile, and optionally a
    nuclei mask. It supports channel selection, preprocessing, filtering, and both spatial and
    color augmentations.
    Args:
        dataframe (pd.DataFrame): Tile-level DataFrame containing file paths and metadata for each
            tile. Must include 'image_path', 'target_path' columns and optionally 'nuclei_path'.
        in_channel_idxs (list or None, optional): Indices of input image channels to select.
            If None, all channels are used.
        targ_channel_idxs (list or None, optional): Indices of target image channels to select.
            If None, all channels are used.
        preprocess_input_fn (Callable, optional): Function to preprocess the input image after
            augmentations.
        preprocess_target_fn (Callable, optional): Function to preprocess the target image after
            augmentations.
        filter_target_fn (Callable, optional): Function to filter or modify the target image before
            augmentations. Mostly not used.
        spatial_augmentations (Callable, optional): Function or transform to apply same spatial
            augmentations to both image and target (and nuclei if present).
        color_augmentations (Callable, optional): Function or transform to apply color
            augmentations to the input image.
        return_nuclei (bool, optional): If True, loads and returns the nuclei mask for each tile
            (requires 'nuclei_path' column in DataFrame).
    Attributes:
        df (pd.DataFrame): The Tile-level DataFrame containing tile metadata.
        in_channel_idxs (list or None): Selected input image channels.
        targ_channel_idxs (list or None): Selected target image channels.
        preprocess_input_fn (Callable or None): Input preprocessing function.
        preprocess_target_fn (Callable or None): Target preprocessing function.
        filter_target_fn (Callable or None): Target filtering function.
        spatial_augmentations (Callable or None): Spatial augmentation function.
        color_augmentations (Callable or None): Color augmentation function.
        return_nuclei (bool): Whether to return nuclei masks.
    Returns:
        dict: A dictionary containing:
            - 'image' (torch.Tensor): The input image tensor (C, H, W).
            - 'target' (torch.Tensor): The target image tensor (C, H, W).
            - 'tile_name' (str): The tile name (stem of input image path).
            - 'nuclei' (torch.Tensor, optional): The nuclei mask tensor, if return_nuclei is True.
            - 'slide_name' (str, optional): The slide name, if 'in_slide_name' is present in the
                DataFrame row.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 in_channel_idxs: List[int] = None,
                 targ_channel_idxs: List[int] = None,
                 preprocess_input_fn: Optional[Callable] = None,
                 preprocess_target_fn: Optional[Callable] = None,
                 filter_target_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 return_nuclei: bool = False,
                 ):
        self.df = dataframe

        self.in_channel_idxs = in_channel_idxs
        self.targ_channel_idxs = targ_channel_idxs

        self.preprocess_input_fn = preprocess_input_fn
        self.preprocess_target_fn = preprocess_target_fn

        self.filter_target_fn = filter_target_fn
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.return_nuclei = return_nuclei

    def __getitem__(self, idx: int) -> dict:
        """Load an image tile and its target tile at idx, optionally with a nuclei mask."""
        row = self.df.iloc[idx]
        input_path = row["image_path"]
        target_path = row["target_path"]
        tile_name = Path(input_path).stem

        image = np.asarray(Image.open(input_path))
        target = pyvips.Image.new_from_file(
            target_path, memory=True, access="sequential")
        if self.targ_channel_idxs is not None:
            target = target[self.targ_channel_idxs].numpy()
        else:
            target = target.numpy()

        output_dict = {}
        if self.return_nuclei:
            nuclei_path = row["nuclei_path"]
            nuclei = pyvips.Image.new_from_file(
                nuclei_path, page=0, access="sequential").numpy()
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(target.shape) == 2:
            target = np.expand_dims(target, axis=-1)
        if self.in_channel_idxs is not None:
            image = image[..., self.in_channel_idxs]

        if image.dtype not in [np.uint8, np.float32]:
            image = np.float32(image)
        if target.dtype not in [np.uint8, np.float32]:
            target = np.float32(target)

        if self.filter_target_fn:
            target = self.filter_target_fn(target)

        if self.spatial_augmentations:
            if self.return_nuclei:
                transformed = self.spatial_augmentations(
                    image=image, image_target=target, nuclei=np.int32(nuclei))
                image, target = transformed["image"], transformed["image_target"]
                nuclei = np.uint32(transformed["nuclei"])
            else:
                transformed = self.spatial_augmentations(image=image, image_target=target)
                image, target = transformed["image"], transformed["image_target"]

        if self.color_augmentations:
            image = self.color_augmentations(image=image)["image"]
            image = np.clip(image, 0, 255)

        if self.preprocess_input_fn:
            image = self.preprocess_input_fn(image)
        if self.preprocess_target_fn:
            target = self.preprocess_target_fn(target)

        if not image.flags.writeable:
            image = image.copy()
        image = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)

        output_dict.update({"image": image, "target": target, "tile_name": tile_name})
        if self.return_nuclei:
            nuclei = torch.from_numpy(nuclei)
            output_dict.update(
                {"nuclei": nuclei})

        if "in_slide_name" in row.keys():
            output_dict["slide_name"] = row["in_slide_name"]
        return output_dict

    def reset(self):
        """Only for compatibility with WSIs datasets."""
        pass

    def __len__(self):
        return len(self.df)


class Img2ImgNucleiSlideDataset(Dataset):
    """
    PyTorch Dataset using WSI SlideVips reader for loading paired image and target tiles from a \
    DataFrame, with optional nuclei masks and augmentations.

    This dataset is designed to work with registered examples from WSIs, where each sample
    consists of a region from an input slide and a corresponding region from a target slide.
    Optionally, a nuclei mask can also be returned for each sample.
    Args:
        slide_dataframe (pd.DataFrame): Slide-level DataFrame with slide-level metadata and
            file paths.
        dataframe (pd.DataFrame): Tile-level DataFrame with sample-level metadata and coordinates.
        in_channel_idxs (list or None, optional): Indices of input channels to use.
            Defaults to None (all channels).
        targ_channel_idxs (list or None, optional): Indices of target channels to use.
            Defaults to None (all channels).
        mode_in (str, optional): Color mode for input slides. Defaults to "RGB".
        mode_targ (str, optional): Color mode for target slides. Defaults to "RGB".
        preprocess_input_fn (Callable or None, optional): Function to preprocess input images.
            Defaults to None.
        preprocess_target_fn (Callable or None, optional): Function to preprocess target images.
            Defaults to None.
        filter_target_fn (Callable or None, optional): Function to filter/modify target images.
            Defaults to None. Mostly not used.
        spatial_augmentations (Callable, optional): Function or transform to apply same spatial
            augmentations to both image and target (and nuclei if present).
        color_augmentations (Callable, optional): Function or transform to apply color
            augmentations to the input image.
        return_nuclei (bool, optional): Whether to return nuclei masks. Defaults to False.
        reiter_fetch (bool, optional): Whether to reinitialize slide objects on each fetch.
            Defaults to False. Deprecated.
    Attributes:
        df (pd.DataFrame): Tile-level DataFrame containing sample metadata and coordinates.
        inslide_name2path (dict): Mapping from input slide names to file paths.
        targslide_name2path (dict): Mapping from target slide names to file paths.
        return_nuclei (bool): Whether to return nuclei masks.
        nucleislide_name2path (dict): Mapping from slide names to nuclei mask file paths
            (if return_nuclei is True).
        nuclei_targ_dict (dict): Cache for loaded nuclei slide objects.
        slide_in_dict (dict): Cache for loaded input slide objects.
        slide_targ_dict (dict): Cache for loaded target slide objects.
        in_channel_idxs (list or None): Indices of input channels to use.
        targ_channel_idxs (list or None): Indices of target channels to use.
        mode_in (str): Color mode for input slides (e.g., "RGB").
        mode_targ (str): Color mode for target slides (e.g., "RGB").
        preprocess_input_fn (Callable or None): Function to preprocess input images.
        preprocess_target_fn (Callable or None): Function to preprocess target images.
        filter_target_fn (Callable or None): Function to filter/modify target images.
        spatial_augmentations (Callable or None): Function for spatial augmentations.
        color_augmentations (Callable or None): Function for color augmentations.
        reiter_fetch (bool): Whether to reinitialize slide objects on each fetch. Deprecated.
    Methods:
        reset(): Clears cached slide objects.
        __getitem__(idx): Returns a sample dictionary with input, target, and optional nuclei mask.
        __len__(): Returns the number of samples in the dataset.
    Returns:
        dict: A dictionary containing:
            - 'image' (torch.Tensor): The input image tensor (C, H, W).
            - 'target' (torch.Tensor): The target image tensor (C, H, W).
            - 'tile_name' (str): The tile name, constructed from slide name, coordinates, level,
                and tile size.
            - 'slide_name' (str): The slide name for this tile.
            - 'location' (tuple): The (x, y) coordinates of the tile in the slide.
            - 'nuclei' (torch.Tensor, optional): The nuclei mask tensor, if return_nuclei is True.
    Warning:
        This dataset from WSI currently generates RAM issues. We recommend using tile datasets
            for now.
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
                 return_nuclei: bool = False,
                 reiter_fetch: bool = False,
                 ):
        #  slide dataframe and dataframe or only one ?
        assert dataframe["in_slide_name"].isin(slide_dataframe["in_slide_name"].tolist()).all()
        slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(
            dataframe["in_slide_name"].unique())]

        self.df = dataframe
        self.inslide_name2path = slide_dataframe.set_index(
            "in_slide_name")["in_slide_path"].to_dict()
        self.targslide_name2path = slide_dataframe.set_index(
            "in_slide_name")["targ_slide_path"].to_dict()
        self.return_nuclei = return_nuclei
        if self.return_nuclei:
            self.nucleislide_name2path = slide_dataframe.set_index(
                "in_slide_name")["nuclei_slide_path"].to_dict()
            self.nuclei_targ_dict = {}

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
        """Reset internal state for DataLoader worker initialization."""
        self.slide_in_dict.clear()
        self.slide_targ_dict.clear()
        if self.return_nuclei:
            self.nuclei_targ_dict.clear()

    def __getitem__(self, idx: int) -> dict:
        """Load a sample at idx, including input image, target image, and optional nuclei mask."""
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
            slide_in.set_concurrency(1)
            self.slide_in_dict[slide_name] = slide_in
        try:
            slide_targ = self.slide_targ_dict[slide_name]
        except KeyError:
            slide_targ = SlideVips(
                self.targslide_name2path[slide_name], self.targ_channel_idxs,
                self.mode_targ, self.reiter_fetch)
            slide_targ.set_concurrency(1)
            self.slide_targ_dict[slide_name] = slide_targ
        if self.return_nuclei:
            try:
                slide_nuclei = self.nuclei_targ_dict[slide_name]
            except KeyError:
                slide_nuclei = SlideVips(
                    self.nucleislide_name2path[slide_name],
                    mode="IF", channel_idxs=[0], reiter_fetch=self.reiter_fetch)
                slide_nuclei.set_concurrency(1)
                self.nuclei_targ_dict[slide_name] = slide_nuclei
            nuclei = slide_nuclei.read_region(location, level, tile_size)

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
            if self.return_nuclei:
                transformed = self.spatial_augmentations(
                    image=image, image_target=target, nuclei=np.int32(nuclei))
                image, target = transformed["image"], transformed["image_target"]
                nuclei = np.uint32(transformed["nuclei"])
            else:
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
        output_dict = {"image": image, "target": target, "tile_name": tile_name}
        if self.return_nuclei:
            nuclei = torch.from_numpy(nuclei)
            output_dict.update(
                {"slide_name": slide_name, "nuclei": nuclei, "location": location})

        return output_dict

    def __len__(self):
        return len(self.df)


class BalancedPositiveSampler(Sampler[int]):  # Can be usefull only when predicting one channel
    """
    A PyTorch Sampler for balancing rare cell types in H&E to mIF translation tasks.

    This sampler addresses class imbalance by oversampling tiles (patches) containing rare cell
    types, as determined by the number of positive cells for a given marker (cell type) in each
    patch. For each patch, the count column (e.g., "CD8_count") indicates the number of positive
    cells for that marker. The sampler selects the marker (cell type) with the most positive
    samples above a threshold, and balances the dataset by oversampling tiles with high counts for
    that marker and undersampling or sampling a fixed proportion of 'other' tiles.

    Args:
        dataframe (pd.DataFrame): Tile-level dataFrame with per-tile cell type counts
            (e.g., "CD8_count").
        class_names (List[str]): List of marker/cell type names (e.g., ["CD8", "CD4", ...]).
        thresh (int): Minimum number of positive cells in a tile to be considered "positive"
            for balancing.
        other_percent (float, optional): Proportion of negative tiles to include.
            Defaults to 0.20.

    Attributes:
        dataframe (pd.DataFrame): Copy of the input DataFrame, reset with new indices.
        total_size (int): Total number of samples in the DataFrame.
        other_percent (float): Proportion of 'other' samples to include.
        column_name (str): The column name corresponding to the selected positive cell type.
        thresh (int): Threshold for positive sample selection.
        indices (np.ndarray): Array of sampled indices for the current epoch.

    Example:
        sampler = BalancedPositiveSampler(df, class_names=["CD8", "CD4"], thresh=5,
                                          other_percent=0.2)

    Warning:
        This sampler works best when there is only one mIF channel (i.e., one class_name).
        Using multiple class_names may lead to suboptimal sampling.
    """

    def __init__(self, dataframe: pd.DataFrame, class_names: List[str], thresh: int,
                 other_percent: float = 0.20):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.total_size = len(self.dataframe)
        self.other_percent = other_percent

        column_names = [f"{class_name}_count" for class_name in class_names]
        if len(class_names) > 1:
            print("Warning: BalancedPositiveSampler is most effective when training with a single "
                  "mIF channel (one class_name). Multiple class_names may lead to suboptimal "
                  "sampling.")
        df_columns = self.dataframe[column_names]
        idx_max = (df_columns > thresh).sum(axis=0).argmax()
        self.column_name = column_names[idx_max]
        assert type(thresh) is int
        assert thresh > 0
        self.thresh = thresh
        self.indices = self.create_indices()

    def create_indices(self):
        """Create and return a shuffled array of sampled indices."""
        df_column = self.dataframe[self.column_name]
        other_indices = self.dataframe[df_column <= self.thresh].index.to_numpy()
        pos_indices = self.dataframe[df_column > self.thresh].index.to_numpy()

        factor_pos = int(self.total_size * (1 - self.other_percent)) / len(pos_indices)
        idxs_positivity_sampled = self.sampling(pos_indices, factor_pos)
        factor_other = int(self.total_size * self.other_percent) / len(other_indices)
        idxs_other_sampled = self.sampling(other_indices, factor_other)
        combined_idxs = np.hstack((idxs_positivity_sampled, idxs_other_sampled))
        np.random.shuffle(combined_idxs)
        print(len(idxs_positivity_sampled), len(idxs_other_sampled))
        return combined_idxs

    def sampling(self, idxs: np.ndarray, factor: float) -> np.ndarray:
        """Sample indices by a given factor, supporting both upsampling and downsampling."""
        if factor <= 0:
            raise ValueError("factor must be greater than 0")
        elif factor == 1:
            return idxs
        elif factor > 1:
            int_factor = int(factor)
            idxs_up = np.repeat(idxs, int_factor)
            factor_residual = factor - int_factor
            idxs_up_res = np.random.choice(idxs, size=int(len(idxs) * factor_residual),
                                           replace=False)
            idxs_sampled = np.hstack((idxs_up, idxs_up_res))
        else:
            idxs_sampled = np.random.choice(idxs, size=int(len(idxs) * factor), replace=False)
        return idxs_sampled

    def __iter__(self):
        self.indices = self.create_indices()
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


class NormalizationLayer:
    """
    Normalization layer for preprocessing and unnormalizing image data.

    This class supports two normalization modes:
    - "he": Standard normalization using provided channel means and standard deviations.
    - "if": mIF intensity feature normalization, scaling values to [-0.9, 0.9] range.
        No need for statistics with this mode.
    Compared to other usual normalization layers, statistics are in original range, i.e.
    between 0 and 255, and not in [0, 1] range.
    Args:
        stats_list (list or dict): List of statistics dictionaries, each containing 'mean' and
            'std' keys, or a single dictionary for one channel.
        mode (str, optional): Normalization mode, either "he" or "if". Defaults to "he".
    Attributes:
        mean (np.ndarray): Mean values for normalization (used in "he" mode).
        std (np.ndarray): Standard deviation values for normalization (used in "he" mode).
        mode (str): Normalization mode, either "he" or "if".
    Methods:
        __call__(x):
            Applies normalization to the input data `x` according to the selected mode.
        unormalize(x):
            Reverts the normalization on the input data `x` according to the selected mode.
    """

    def __init__(self, stats: Optional[dict] = None, mode: str = "he"):
        assert mode in ["he", "if"]
        if mode == "he":
            if stats is None:
                raise ValueError("stats must be provided for 'he' normalization mode.")
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
            self.mean = np.float32(mean.reshape((1, 1, -1)))
            self.std = np.float32(std.reshape((1, 1, -1)))
            print(mode, mean, std)

        self.mode = mode

    def unormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverses normalization on input tensor x based on the current mode. WARNING: Output is \
        in float32."""
        if self.mode == "if":
            x_unorm = (x + 0.9) * 255 / 1.8
        else:
            x_unorm = x * self.std + self.mean
        return x_unorm

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize input array based on the selected modality."""
        if self.mode == "he":
            x_norm = (x - self.mean) / self.std
        else:
            x_norm = np.float32(x) / 255 * 1.8 - 0.9

        return x_norm
