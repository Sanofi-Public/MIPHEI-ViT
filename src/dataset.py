"""
Dataset classes for image to image segmentation.
"""

from typing import Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import pyvips
from slidevips import SlideVips
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .augmentations import HedColorAugmentor


class DataModule:
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
        else:
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
        return self.train_dataloader, self.val_dataloader, self.test_dataloader


class TileSlideDataset(Dataset): # also from slide directly

    def __init__(self,
                 dataframe: pd.DataFrame,
                 channel_idxs=None,
                 preprocess_input_fn: Optional[Callable] = None,
                 spatial_augmentations = None,
                 color_augmentations = None,
                 return_nuclei=False,
                 ):
        self.df = dataframe

        self.channel_idxs = channel_idxs

        self.preprocess_input_fn = preprocess_input_fn

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.return_nuclei = return_nuclei

    def __getitem__(self, idx):
        # load images and target
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
        pass

    def __len__(self):
        return len(self.df)


class TileImg2ImgSlideDataset(Dataset): # also from slide directly

    def __init__(self,
                 dataframe: pd.DataFrame,
                 in_channel_idxs=None,
                 targ_channel_idxs=None,
                 preprocess_input_fn: Optional[Callable] = None,
                 preprocess_target_fn: Optional[Callable] = None,
                 filter_target_fn: Optional[Callable] = None,
                 spatial_augmentations = None,
                 color_augmentations = None,
                 return_nuclei=False,
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

    def __getitem__(self, idx):
        # load images and target
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
        pass

    def __len__(self):
        return len(self.df)


class Img2ImgNucleiSlideDataset(Dataset): # also from slide directly

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
                 return_nuclei=False,
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
        self.slide_in_dict.clear()
        self.slide_targ_dict.clear()
        if self.return_nuclei:
            self.nuclei_targ_dict.clear()

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
        if self.return_nuclei:
            try:
                slide_nuclei = self.nuclei_targ_dict[slide_name]
            except KeyError:
                slide_nuclei = SlideVips(
                    self.nucleislide_name2path[slide_name],
                    mode="IF", channel_idxs=[0], reiter_fetch=self.reiter_fetch)
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


def get_augmentations(width, height, return_nuclei=False, training=True):
    additional_targets = {'image_target': 'image'}
    if return_nuclei:
        additional_targets["nuclei"] = "image"
    if training:
        spatial_augmentations = A.Compose([
            A.RandomCrop(width=width, height=height),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(p=0.1, num_holes_range=[1, 1], hole_height_range=[0., 0.3], hole_width_range=[0., 0.3])
        ], additional_targets=additional_targets)

        color_augmentations = A.Compose([
            HedColorAugmentor(thresh=0.015, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), ##
            A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 1.5), p=0.1),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.1)
        ])
    else:
        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width, height=height),
        ], additional_targets=additional_targets)

        color_augmentations = None

    return spatial_augmentations, color_augmentations


class BalancedPositiveSampler(Sampler[int]):
    def __init__(self, dataframe, class_names, thresh, other_percent=0.20):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.total_size = len(self.dataframe)
        self.other_percent = other_percent

        column_names = [f"{class_name}_count" for class_name in class_names]
        df_columns = self.dataframe[column_names]
        idx_max = (df_columns > thresh).sum(axis=0).argmax()
        self.column_name = column_names[idx_max]
        assert type(thresh) == int
        assert thresh > 0
        self.thresh = thresh
        self.indices = self.create_indices()

    def create_indices(self):
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

    def sampling(self, idxs, factor):
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


def dataloader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


class NormalizationLayer:

    def __init__(self, stats_list, mode="he"):
        assert mode in ["he", "if"]
        if mode == "he":
            if not isinstance(stats_list, list):
                stats_list = [stats_list].copy()
            else:
                stats_list = stats_list.copy()
            mean = np.array([stats["mean"] for stats in stats_list])
            std = np.array([stats["std"] for stats in stats_list])
            self.mean = np.float32(mean.reshape((1, 1, -1)))
            self.std = np.float32(std.reshape((1, 1, -1)))
            print(mode, mean, std)

        self.mode = mode

    def unormalize(self, x):
        if self.mode == "if":
            x_unorm = (x + 0.9) * 255 / 1.8
        else:
            x_unorm = x * self.std + self.mean
        return x_unorm

    def __call__(self, x):
        if self.mode == "he":
            x_norm = (x - self.mean) / self.std
        else:
            x_norm = np.float32(x) / 255 * 1.8 - 0.9

        return x_norm


def get_width_height(dataframe):
    from_slide = "image_path" not in dataframe.columns
    if from_slide:
        width = dataframe["tile_size_x"].iloc[0]
        height = dataframe["tile_size_y"].iloc[0]
    else:
        width, height = Image.open(dataframe["image_path"].iloc[0]).size

    return width, height

def get_effective_width_height(width, height, train=False):
    if train:
        # Calculate the largest power of 2 less than or equal to width and height
        width = int(2 ** (np.floor(np.log2(width))))
        height = int(2 ** (np.floor(np.log2(height))))

    return width, height

def get_input_mean_std(cfg, channel_stats_rgb):
    if cfg.model.model_name in ["cellvit", "vitmatte"]:
        mean, std = np.asarray([0.485, 0.456, 0.406])*255, np.asarray([0.229, 0.224, 0.225])*255
    elif cfg.model.model_name.startswith("unet") or cfg.model.model_name.startswith("myvitmatte"):
        if cfg.model.encoder.encoder_name == "hoptimus0":
            mean, std = np.asarray([0.707223, 0.578729, 0.703617])*255, np.asarray([0.211883, 0.230117, 0.177517])*255
        else:
            mean, std = np.asarray([0.485, 0.456, 0.406])*255, np.asarray([0.229, 0.224, 0.225])*255
    else:
        mean, std = channel_stats_rgb["mean"], channel_stats_rgb["std"]
    return {"mean": mean, "std": std}
