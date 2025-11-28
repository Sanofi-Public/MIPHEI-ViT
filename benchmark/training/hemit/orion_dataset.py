import os
import pyvips
import torch
from data.base_dataset import BaseDataset
import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image


def normalize(x, **kwargs):
    return x / np.float32(127.5) - np.float32(1.0)

def to_tensor(x, **kwargs):
    return torch.from_numpy(x).permute(2, 0, 1)


def get_transform_orion(opt):
    transform_list = []

    if "resize" in opt.preprocess:
        transform_list.append(
            A.Resize(opt.load_size, opt.load_size)
        )

    if "crop" in opt.preprocess:
        if opt.phase == "train":
            transform_list.append(
                A.RandomCrop(opt.crop_size, opt.crop_size)
            )
        else:
            transform_list.append(
                A.CenterCrop(opt.crop_size, opt.crop_size)
            )

    if not opt.no_flip:
        transform_list += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]

    transform_list.append(
            A.Lambda(normalize)
        )

    transform_list.append(A.Lambda(to_tensor))

    return A.Compose(transform_list, additional_targets={
        "target": "image",})


class OrionDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.df = pd.read_csv(os.path.join(opt.dataroot, f"{opt.phase}_dataframe.csv"))  # get the image directory
        if opt.max_dataset_size and opt.max_dataset_size != float("inf"):
            self.df = self.df.sample(opt.max_dataset_size, random_state=42).reset_index(drop=True)

        assert self.opt.load_size >= self.opt.crop_size  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

        self.transform = get_transform_orion(self.opt)

        self.channel_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        row = self.df.iloc[index]
        A_path = row["image_path"]
        A = np.asarray(Image.open(A_path).convert("RGB"))
        B_path = row["target_path"]
        B = pyvips.Image.new_from_file(B_path)[self.channel_idxs].numpy()

        transformed = self.transform(image=A, target=B)
        A = transformed["image"]
        B = transformed["target"]

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.df)
