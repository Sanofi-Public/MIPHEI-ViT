import pyvips
import os
from data.base_dataset import BaseDataset, get_params, get_transformA, get_transformB
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset in different folders.

    It assumes that the directory '/path/to/data/train' contains two subfolders: trainA, trainB
    During test time, you need to prepare a directory '/path/to/data/test' which also contains testA, testB
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        #self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path)
        B_img = pyvips.Image.new_from_file(B_path)
        
        B_img = B_img.crop(38, 38, 256, 256).numpy()
        B_img = np.delete(B_img, 13, axis=-1)
        A_img = np.asarray(A_img.crop((38, 38, 294, 294)))

        B = (torch.from_numpy(B_img.copy()).permute((2, 0, 1)).float() / 255 - 0.5) / 0.5
        A = (torch.from_numpy(A_img.copy()).permute((2, 0, 1)).float() / 255 - 0.5) / 0.5

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        #return {A, B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        assert len(self.A_paths) == len(self.B_paths)
        return len(self.A_paths)
