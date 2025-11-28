import pyvips
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image
import tifffile
import pandas as pd
import torch
from torch.utils.data import Dataset

from slidevips import SlideVips
from slidevips.reader import NUMPY_DTYPE_MAPPING


def get_png_pyramid_pyvips(filepath: str, channel_idxs: Optional[List[int]] = None):

    image_array = np.asarray(Image.open(filepath))
    image = pyvips.Image.new_from_array(image_array)

    if channel_idxs is not None:
        image = image[channel_idxs]

    return [image]


class SlideVipsPNG(SlideVips):
    def __init__(self, filepath: str, mpp: float, channel_idxs: List[int] = None,
                 reiter_fetch: bool = False):
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0)
        pyvips.cache_set_max_files(0)
        assert os.path.exists(filepath)
        self.pyramid_image = get_png_pyramid_pyvips(filepath, channel_idxs)
        self.fields = {"mpp": mpp}
        self.level_count = len(self.pyramid_image)
        self.mpp = mpp
        self.compute_spatial_attributes(self.mpp)

        self.slide_name = Path(filepath).stem
        self.n_channels = self.pyramid_image[0].bands

        self.dtype = self.pyramid_image[0].get("format")
        self.dtype_numpy = NUMPY_DTYPE_MAPPING[self.dtype]
        self._reiter_fetch = reiter_fetch


class LizardDataset(Dataset):

    def __init__(self,
                 slide_dataframe: pd.DataFrame,
                 dataframe: pd.DataFrame,
                 mpp: float,
                 preprocess_input_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 return_nuclei: bool = False,
                 ):
        #  slide dataframe and dataframe or only one ?
        assert dataframe["in_slide_name"].isin(slide_dataframe["in_slide_name"].tolist()).all()
        slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(
            dataframe["in_slide_name"].unique())]

        self.df = dataframe
        self.inslide_name2path = slide_dataframe.set_index(
            "in_slide_name")["in_slide_path"].to_dict()
        self.return_nuclei = return_nuclei
        if self.return_nuclei:
            self.nucleislide_name2path = slide_dataframe.set_index(
                "in_slide_name")["nuclei_slide_path"].to_dict()
            self.nuclei_targ_dict = {}

        self.slide_in_dict = {}

        self.preprocess_input_fn = preprocess_input_fn
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.mpp = mpp

    def reset(self):
        """Reset internal state for DataLoader worker initialization."""
        self.slide_in_dict.clear()
        if self.return_nuclei:
            self.nuclei_targ_dict.clear()

    def __getitem__(self, idx: int) -> dict:
        """Load a sample at idx, including input image and optional nuclei mask."""
        row = self.df.iloc[idx]
        slide_name = row["in_slide_name"]
        location = (row["x"], row["y"])
        level = row["level"]
        tile_size = (row["tile_size_x"], row["tile_size_y"])
        tile_name = "_".join(map(str, [slide_name, *location, level, *tile_size]))

        try:
            slide_in = self.slide_in_dict[slide_name]
        except KeyError:
            slide_in = SlideVipsPNG(
                self.inslide_name2path[slide_name],
                mpp=self.mpp)
            slide_in.set_concurrency(1)
            self.slide_in_dict[slide_name] = slide_in
        if self.return_nuclei:
            try:
                slide_nuclei = self.nuclei_targ_dict[slide_name]
            except KeyError:
                slide_nuclei = SlideVipsPNG(
                    self.nucleislide_name2path[slide_name], channel_idxs=[0],
                    mpp=self.mpp)
                slide_nuclei.set_concurrency(1)
                self.nuclei_targ_dict[slide_name] = slide_nuclei
            nuclei = slide_nuclei.read_region(location, level, tile_size)

        image = slide_in.read_region(location, level, tile_size)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

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

        image = torch.from_numpy(image).permute(2, 0, 1)
        output_dict = {"image": image, "tile_name": tile_name}
        if self.return_nuclei:
            nuclei = torch.from_numpy(nuclei)
            output_dict.update(
                {"slide_name": slide_name, "nuclei": nuclei, "location": location})

        return output_dict

    def __len__(self):
        return len(self.df)


class TileDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 preprocess_input_fn: Optional[Callable] = None,
                 spatial_augmentations=None,
                 color_augmentations=None,
                 return_nuclei: bool = False,
                 ):
        self.df = dataframe

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
                nuclei_path).numpy()
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

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


class BaseOutputRosieDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 preprocess_pred_fn: Optional[Callable] = None,
                 preprocess_target_fn: Optional[Callable] = None,
                 pred_channel_idxs: Optional[List[int]] = None,
                 targ_channel_idxs: Optional[List[int]] = None,
                 spatial_augmentations=None,
                 pred_augmentations=None,
                 return_target: bool = True,
                 return_nuclei: bool = False,
                 ):
        self.df = dataframe

        self.preprocess_pred_fn = preprocess_pred_fn
        self.preprocess_target_fn = preprocess_target_fn

        self.pred_channel_idxs = pred_channel_idxs
        self.targ_channel_idxs = targ_channel_idxs

        self.spatial_augmentations = spatial_augmentations
        self.pred_augmentations = pred_augmentations

        self.return_target = return_target
        self.return_nuclei = return_nuclei


    def read_pred(self, row):
        pred_path = row["pred_path"]

        pred = tifffile.imread(pred_path)
        if self.pred_channel_idxs is not None:
            pred = pred[..., self.pred_channel_idxs]

        return pred

    def read_target(self, row):
        raise NotImplementedError

    def read_nuclei(self, row):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Load an image tile at idx and optionally its nuclei mask."""
        row = self.df.iloc[idx]
        pred_path = row["pred_path"]
        tile_name = Path(pred_path).stem

        pred = self.read_pred(row)
        if len(pred.shape) == 2:
            pred = np.expand_dims(pred, axis=-1)
        if self.pred_augmentations:
            pred = self.pred_augmentations(image=pred)["image"]
        spatial_aug_inputs = {"image": pred}

        if self.return_target:
            target = self.read_target(row)
            if len(target.shape) == 2:
                target = np.expand_dims(target, axis=-1)
            spatial_aug_inputs["image_target"] = target

        if self.return_nuclei:
            nuclei = self.read_nuclei(row)
            spatial_aug_inputs["nuclei"] = nuclei

        if self.spatial_augmentations:
            transformed = self.spatial_augmentations(**spatial_aug_inputs)
            pred = transformed["image"]
            if self.return_nuclei:
                nuclei = np.uint32(transformed["nuclei"])
            if self.return_target:
                target = transformed["image_target"]

        if self.preprocess_pred_fn:
            pred = self.preprocess_pred_fn(pred)

        if self.preprocess_target_fn and self.return_target:
            target = self.preprocess_target_fn(target)

        pred = torch.from_numpy(pred).permute(2, 0, 1)

        output_dict = {"pred": pred, "tile_name": tile_name}
        if self.return_target:
            target = torch.from_numpy(target).permute(2, 0, 1)
            output_dict["target"] = target
        if self.return_nuclei:
            nuclei = torch.from_numpy(nuclei)
            output_dict["nuclei"] = nuclei

        if "in_slide_name" in row.keys():
            output_dict["slide_name"] = row["in_slide_name"]
        return output_dict

    def reset(self):
        """Only for compatibility with WSIs datasets."""
        pass

    def __len__(self):
        return len(self.df)


class TileOutputRosieDataset(BaseOutputRosieDataset):

    def read_target(self, row):
        target_path = row["target_path"]
        target = pyvips.Image.new_from_file(target_path)
        if self.targ_channel_idxs:
            target = target[self.targ_channel_idxs]
        target = target.numpy()
        return target

    def read_nuclei(self, row):
        nuclei_path = row["nuclei_path"]
        nuclei = pyvips.Image.new_from_file(
            nuclei_path)[0].numpy()
        return nuclei


class SlideOutputRosieDataset(BaseOutputRosieDataset):
    def __init__(self, slide_dataframe, wsi_reader_cls=SlideVips, wsi_reader_kwargs: Optional[Dict] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.df["in_slide_name"].isin(slide_dataframe["in_slide_name"].tolist()).all()
        slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(
            self.df["in_slide_name"].unique())]
        self._wsi_reader_cls = wsi_reader_cls
        self._wsi_reader_kwargs = wsi_reader_kwargs if wsi_reader_kwargs is not None else {}

        if self.return_target:
            self.targ_slide_name2path = slide_dataframe.set_index("in_slide_name")["targ_slide_path"].to_dict()
            self.slide_targ_dict = {}
        if self.return_nuclei:
            self.nuclei_slide_name2path = slide_dataframe.set_index("in_slide_name")["nuclei_slide_path"].to_dict()
            self.slide_nuclei_dict = {}

    def read_target(self, row):
        slide_name = row["in_slide_name"]
        location = (row["x"], row["y"])
        level = row["level"]
        tile_size = (row["tile_size_x"], row["tile_size_y"])
        try:
            slide_targ = self.slide_targ_dict[slide_name]
        except KeyError:
            slide_targ = self._wsi_reader_cls(
                self.targ_slide_name2path[slide_name], self.targ_channel_idxs, **self._wsi_reader_kwargs)
            slide_targ.set_concurrency(1)
            self.slide_targ_dict[slide_name] = slide_targ
        target = slide_targ.read_region(location, level, tile_size)
        return target

    def read_nuclei(self, row):
        slide_name = row["in_slide_name"]
        location = (row["x"], row["y"])
        level = row["level"]
        tile_size = (row["tile_size_x"], row["tile_size_y"])
        try:
            slide_nuclei = self.slide_nuclei_dict[slide_name]
        except KeyError:
            #  SlideVips: IF for non RGB channel, channel 0 for nuclei
            slide_nuclei = self._wsi_reader_cls(
                self.nuclei_slide_name2path[slide_name], **self._wsi_reader_kwargs)
            slide_nuclei.set_concurrency(1)
            self.slide_nuclei_dict[slide_name] = slide_nuclei
        nuclei = slide_nuclei.read_region(location, level, tile_size)
        return nuclei
