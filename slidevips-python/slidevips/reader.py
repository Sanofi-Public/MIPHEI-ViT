import concurrent
import concurrent.futures
import cv2
import numpy as np
import os
from pathlib import Path
from numpy import ndarray
import pyvips
from slidevips.read_pyramid import get_pyramid_pyvips
import torch
from typing import List, Tuple, Optional

import gc


NUMPY_DTYPE_MAPPING = {
    "uchar": np.uint8,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32
}


class SlideVips:
    """
    A class representing a slide in the SlideVips library. Native implementation
        that uses default crop function.

    Args:
        filepath (str): The path to the slide file.
        channel_idxs (list, optional): List of channel indexes to keep. Defaults to None.
        mode (str, optional): The mode of the slide. Defaults to "RGB".
        reiter_fetch (bool): Try to load the tile until no Error. Usefull when data is on nfs.

    Attributes:
        pyramid_image (list): List of pyramid images.
        fields (dict): Dictionary containing slide fields.
        level_count (int): The number of levels in the slide.
        magnification (int): The magnification of the slide.
        dimensions (tuple): The dimensions of the slide.
        level_dimensions (numpy.ndarray): Array of level dimensions.
        level_downsamples (numpy.ndarray): Array of level downsamples.
        level_magnifications (numpy.ndarray): Array of level magnifications.
        downsample (float): The downsample of the slide.
        mode (str): The mode of the slide.
        slide_name (str): The name of the slide.
        dtype (str): The data type of the slide.
        dtype_numpy (str): The corresponding numpy data type.
        _reiter_fetch (bool): If we persist to load a tile if there is an error.

    Methods:
        read_region: Read a region from the slide.
        read_region_torch: Read a region from the slide and convert it to a torch tensor.
        read_regions: Read multiple regions from the slide.
        write_region: Write a region from the slide to a file.
        write_regions: Write multiple regions from the slide to files.
        get_thumbnail: Get a thumbnail of the slide.
        close: Close the slide.
    """

    def __init__(self, filepath: str, channel_idxs: List[int] = None, mode: str = "RGB",
                 reiter_fetch: bool = False) -> None:
        """
        Initializes a Reader object.

        Args:
            filepath (str): The path to the image file.
            channel_idxs (list, optional): List of channel indices to keep. Defaults to None.
            mode (str, optional): The mode of the image. Defaults to "RGB".
        """
        # keep only some channels
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0)
        pyvips.cache_set_max_files(0)
        assert os.path.exists(filepath)
        self.pyramid_image, self.fields = get_pyramid_pyvips(filepath, channel_idxs, mode)
        self.level_count = len(self.pyramid_image)
        self.mpp = self.fields["mpp"]
        self.compute_spatial_attributes(self.mpp)

        self.slide_name = Path(filepath).stem
        self.n_channels = self.pyramid_image[0].bands

        if self.n_channels in [3, 4]:  # TO DO: deplace in get_pyramid_pyvips
            self.mode = mode
        else:
            self.mode = "IF"
            if self.mode != mode:
                print("Warning: mode identified is IF, but specified as RGB.")

        self.dtype = self.pyramid_image[0].get("format")
        self.dtype_numpy = NUMPY_DTYPE_MAPPING[self.dtype]
        self._reiter_fetch = reiter_fetch

    def compute_spatial_attributes(self, mpp):
        """
        Blabla
        """
        self.absolute_magnification, self.magnification = calculate_magnification(mpp)
        self.dimensions = np.array([self.pyramid_image[0].width, self.pyramid_image[0].height])
        self.level_dimensions = np.array([(image.width, image.height) \
                                            for image in self.pyramid_image])
        self.level_factor = self.level_dimensions / self.dimensions
        self.level_downsamples = np.mean(1 / self.level_factor, axis=1) # same as openslide
        self.level_magnifications = self.magnification * self.level_dimensions / \
            self.dimensions
        self.level_resolutions = mpp * self.level_downsamples

    def resize(self, scale_factor: None):
        """
        Blabla
        """
        for level, image in enumerate(self.pyramid_image):
            self.pyramid_image[level] = image.resize(scale_factor)
        self.mpp = self.mpp / scale_factor
        self.compute_spatial_attributes(self.mpp)

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> np.ndarray:
        """
        Read a region from the slide at the specified location, level, and size.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.

        Returns:
            numpy.ndarray: The region as a NumPy array.
        """
        region = self._pyvips_crop_region(location, level, size).numpy()
        region_array = region
        #region_array = pyvips2numpy(region, self.dtype_numpy)

        return region_array

    def read_region_torch(self, location: Tuple[int, int], level: int,
                          size: Tuple[int, int]) -> torch.Tensor:
        """
        Reads a region from the image using pyvips and converts it to a torch tensor.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.

        Returns:
            torch.Tensor: The region as a torch tensor.
        """
        region = self._pyvips_crop_region(location, level, size)
        region_torch = pyvips2torch(region, self.dtype_numpy)
        return region_torch

    def read_regions(self, locations: List[Tuple[int, int]], level: int, size: int) -> np.ndarray:
        """
        Reads regions from the specified locations in parallel using ThreadPoolExecutor.

        Args:
            locations (list): List of locations to read regions from.
            level (int): Level of the regions to read.
            size (int): Size of the regions to read.

        Returns:
            numpy.ndarray: Array of regions read from the specified locations.
        """
        # Initialize an empty list for regions with the size of locations
        regions = [None] * len(locations)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each location and store the futures with their index
            future_to_index = {executor.submit(self.read_region, location, level, size): i for i, location in enumerate(locations)}

            # Collect the results as they become available, using the index to place them correctly
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    region = future.result()
                    regions[index] = region  # Place the region at the correct index
                except Exception as e:
                    location = locations[index]
                    print(f"Error occurred while reading region at location {location}: {e}")

        # Stack the regions along the first axis
        regions = np.stack(regions, axis=0)
        return regions

    def write_region(self, folder: str, location: Tuple[int, int], level: int,
                     size: Tuple[int, int], img_format: str = ".png",
                     filename: Optional[str] = None) -> str:
        """
        Writes a region of the slide to an image file.

        Args:
            folder (str): The folder where the image file will be saved.
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.
            img_format (str, optional): The image format of the output file. Defaults to ".png".
            filename (str, optional): The name of the output file. If not provided, a filename
                will be generated based on the slide name, location, level, size, and image format.

        Returns:
            str: The path to the saved image file.
        """
        if filename is None:
            image_path = Path(folder) / "{}_{}_{}_{}_{}_{}{}".format(
            self.slide_name, *location, level, *size, img_format)
        else:
            image_path = Path(folder) / filename
        region = self._pyvips_crop_region(location, level, size)
        region.write_to_file(image_path)
        return image_path

    def write_regions(self, folder: str, locations: List[Tuple[int, int]], level: int,
                      size: int, img_format: str = ".png") -> List[str]:
        """
        Write regions of an image to files in the specified folder in parallel
            using ThreadPoolExecutor.

        Args:
            folder (str): The folder path where the image regions will be saved.
            locations (list): A list of locations specifying the regions to be written.
            level (int): The zoom level of the image.
            size (int): The size of the image regions.
            img_format (str, optional): The format of the saved image files. Defaults to ".png".

        Returns:
            list: A list of paths to the saved image files.
        """
        image_paths = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each location
            future_to_location = {executor.submit(
                self.write_region, folder, location,
                level, size, img_format): location for location in locations}

            # Collect the results as they become available
            for future in concurrent.futures.as_completed(future_to_location):
                location = future_to_location[future]
                try:
                    image_path = future.result()
                    image_paths.append(image_path)
                except Exception as e:
                    print(f"Error occurred while writing region at location {location}: {e}")
        return image_paths

    def _pyvips_crop_region(self, location: Tuple[int, int], level: int,
                            size: Tuple[int, int]) -> np.ndarray:
        """
        Crop a region from the pyramid image at the specified level.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The level of the pyramid image to crop from.
            size (tuple): The width and height of the region.

        Returns:
            pyvips.Image: The cropped region.
        """
        image = self.pyramid_image[level]
        location_level = location * self.level_factor[level]

        left, top = location_level
        left, top = int(left), int(top)
        width, height = size

        # Check if the region is completely within the image boundaries
        if (0 <= left < image.width and 0 <= top < image.height and
            left + width <= image.width and top + height <= image.height):
            region = pyvips_fetch_region(image, left, top, width,
                                         height, self._reiter_fetch)
        else:
            # Calculate the region to extract
            region_left = max(left, 0)
            region_top = max(top, 0)
            region_right = min(left + width, image.width)
            region_bottom = min(top + height, image.height)

            region_width = region_right - region_left
            region_height = region_bottom - region_top

            # Create a black image and insert the cropped region into it
            black_image = pyvips.Image.black(width, height, bands=image.bands).cast(image.format)

            if region_width > 0 and region_height > 0:
                region = pyvips_fetch_region(image, region_left, region_top, region_width,
                                             region_height, self._reiter_fetch)

                # Calculate the position to insert the region into the black image
                insert_left = max(0, -left)
                insert_top = max(0, -top)

                # Insert the region into the black image
                black_image = black_image.insert(region, insert_left, insert_top)

            region = black_image

        return region

    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Get a thumbnail of the image.

        Args:
            size: A tuple representing the desired width and height of the thumbnail.

        Returns:
            thumbnail_array: A numpy array representing the thumbnail image.
        """
        level_thumbnail = np.linalg.norm(np.array(size) - self.level_dimensions, axis=1).argmin()
        image = self.pyramid_image[level_thumbnail]
        scale_width = size[0] / image.width
        scale_height = size[1] / image.height
        scale_factor = min(scale_width, scale_height)
        if scale_factor < 0.1:
            raise ValueError("Thumbnail computation: need low res pyramid level")
        thumbnail = image.resize(scale_factor)
        thumbnail_array = pyvips2numpy(thumbnail, self.dtype_numpy)
        return thumbnail_array

    def prune_pyramid(self, level: int) -> None:
        """
        Close each pyramid image and replace them with None except for the specified level.

        Args:
            level (int): The level to prune the pyramid image to.
        """
        if hasattr(self, "pyramid_image"):
            if self.pyramid_image:
                for idx_level, image in enumerate(self.pyramid_image):
                    if idx_level != level:
                        del image
                        self.pyramid_image[idx_level] = None
        else:
            raise ValueError("Pyramid image does not exist")

    def close(self) -> None:
        """
        Closes the reader and clears the pyramid image if it exists.
        """
        if hasattr(self, "pyramid_image"):
            if self.pyramid_image:
                for image in self.pyramid_image:
                    del image
                self.pyramid_image.clear()
            gc.collect()

    def __del__(self):
        """
        Destructor method that closes the reader.
        """
        self.close()


def pyvips_fetch_region(image, left, top, width, height, reiter_fetch=False):
    """
    Helper function to fetch a region from the image.

    Args:
        image (pyvips.Image): The image to crop from.
        left (int): The left coordinate of the region.
        top (int): The top coordinate of the region.
        width (int): The width of the region.
        height (int): The height of the region.

    Returns:
        pyvips.Image: The cropped region.
    """
    if reiter_fetch:
        stop = False
        while not stop:
            try:
                region = image.crop(left, top, width, height)
                stop = True
            except pyvips.Error:
                continue
    else:
        region = image.crop(left, top, width, height)
    return region


class RegionSlideVips(SlideVips):
    """
    A class representing a slide in the SlideVips library using the Region
        and fetch access to the tiles.

    Args:
        filepath (str): The path to the slide image file.
        channel_idxs (list, optional): List of channel indexes to read. Defaults to None.
        mode (str, optional): The mode to read the image in. Defaults to "RGB".
    """

    def __init__(self, filepath: str, channel_idxs=None, mode="RGB", reiter_fetch=False) -> None:
        super().__init__(filepath, channel_idxs, mode, reiter_fetch)
        self.pyramid_region = [pyvips.Region.new(image) for image in self.pyramid_image]

    def read_region(self, location, level: int, size) -> np.ndarray:
        """
        Reads a region from the image using pyvips region.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.

        Returns:
            np.ndarray: The region as a numpy array.
        """
        region_array = self._read_region_wrapper(location, level, size,
                                                 self.dtype_numpy, mode="numpy")
        return region_array

    def read_region_torch(self, location: Tuple[int, int], level: int,
                          size: Tuple[int, int]) -> torch.Tensor:
        """
        Reads a region from the image using pyvips region and converts it to a torch tensor.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.

        Returns:
            torch.Tensor: The region as a torch tensor.
        """
        if self.dtype_numpy == np.uint16:
            region_array = self.read_region(location, level, size)
            region_array = region_array.astype(np.int32)
            region_torch = torch.from_numpy(region_array)
        elif self.dtype_numpy == np.uint8:
            region_torch = self._read_region_wrapper(location, level, size,
                                                     torch.uint8, mode="torch")
        else:
            raise NotImplementedError
        region_torch = torch.permute(region_torch, (2, 0, 1))
        return region_torch

    def _read_region_wrapper(self, location: Tuple[int, int], level: int,
                             size: Tuple[int, int], dtype, mode: str) -> np.ndarray:
        """
        Wrapper function for reading a region from the image.

        Args:
            location (tuple): The top-left coordinates of the region.
            level (int): The zoom level of the region.
            size (tuple): The width and height of the region.
            dtype: The data type of the region.
            mode (str): The mode of reading the region. Should be either "numpy" or "torch".

        Returns:
            np.ndarray: The region as a numpy array.
        """
        region_level = self.pyramid_region[level]
        #image_level = self.pyramid_image[level]
        location_level = location * self.level_factor[level]

        # Image dimensions at the current level
        image_width, image_height = self.level_dimensions[level]

        # Calculate the boundaries of the requested region
        start_x, start_y = int(location_level[0]), int(location_level[1])
        end_x, end_y = start_x + size[0], start_y + size[1]

        # Check if the requested region is entirely outside the image boundaries
        if start_x >= image_width or start_y >= image_height or end_x <= 0 or end_y <= 0:
            raise ValueError("Requested region is outside the slide boundaries.")

        # Determine if padding is needed
        pad_left = -min(start_x, 0)
        pad_top = -min(start_y, 0)
        pad_right = max(end_x - image_width, 0)
        pad_bottom = max(end_y - image_height, 0)

        # Adjust start and end if they are out of bounds
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        end_x = min(end_x, image_width)
        end_y = min(end_y, image_height)

        # Fetch the region
        if self._reiter_fetch:
            stop = False
            while not stop:
                try:
                    buffer = region_level.fetch(start_x, start_y, end_x - start_x, end_y - start_y)
                    stop = True
                except pyvips.Error:
                    continue
        else:
            buffer = region_level.fetch(start_x, start_y, end_x - start_x, end_y - start_y)
        if mode == "torch":
            fetched_region = torch.frombuffer(buffer, dtype=torch.uint8).reshape(
                end_y - start_y, end_x - start_x, self.n_channels)
        else:
            fetched_region = np.ndarray(buffer=buffer, dtype=dtype,
                                        shape=[end_y - start_y, end_x - start_x, self.n_channels])
        #pyvips.vips_lib.vips_region_invalidate(region_level.pointer)
        #image_level.invalidate()

        # Apply padding if needed
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            # Create an array with padding
            if mode == "torch":
                padded_region = torch.zeros((size[1], size[0], self.n_channels), dtype=dtype)
            else:
                padded_region = np.zeros((size[1], size[0], self.n_channels), dtype=dtype)
            padded_region[pad_top: pad_top + fetched_region.shape[0],
                          pad_left: pad_left + fetched_region.shape[1]] = fetched_region
            return padded_region
        else:
            # Return the fetched region directly if no padding is needed
            return fetched_region

    def get_thumbnail(self, size: Tuple[int, int]) -> ndarray:
        idx_dim = np.abs(self.dimensions - np.asarray(size)).argmax()

        scaling = size[idx_dim] / self.dimensions[idx_dim]

        thumbnail_size = np.int32(np.round(self.dimensions * scaling))

        level_dims = self.level_dimensions.copy()
        level_dims = level_dims[np.all(level_dims > thumbnail_size, axis=1)]
        level_thumbnail = np.linalg.norm(thumbnail_size - level_dims, axis=1).argmin()
        level_dim = self.level_dimensions[level_thumbnail]
        if level_dim[0] * level_dim[1] > 225000000:
            raise ValueError("The thumbnail is too large to be processed.")
        image = self.pyramid_region[level_thumbnail]
        buffer = image.fetch(0, 0, level_dim[0], level_dim[1])
        thumbnail = np.ndarray(buffer=buffer, dtype=self.dtype_numpy,
                                    shape=[level_dim[1], level_dim[0], self.n_channels])
        thumbnail = cv2.resize(thumbnail, tuple(thumbnail_size), cv2.INTER_LINEAR)
        return thumbnail


def pyvips2numpy(pyvips_image, dtype):
    """
    Convert a PyVips image to a NumPy array.

    Args:
        pyvips_image (pyvips.Image): The PyVips image to convert.
        dtype (str): The corresponding numpy data type of the pyvips image.

    Returns:
        np.ndarray: The converted NumPy array.

    """
    np_array = np.ndarray(buffer=pyvips_image.write_to_memory(),
                          dtype=dtype,
                          shape=[pyvips_image.height, pyvips_image.width, pyvips_image.bands])
    return np_array


def pyvips2torch(pyvips_image, dtype):
    """
    Convert a PyVips image to a Torch tensor.

    Args:
        pyvips_image (pyvips.Image): The PyVips image to convert.
        dtype (numpy.dtype): The corresponding numpy data type of the pyvips image.

    Returns:
        torch.Tensor: The converted Torch tensor.

    Raises:
        NotImplementedError: If the specified data type is not supported.

    """
    if dtype == np.uint16:
        np_array = pyvips2numpy(pyvips_image, dtype)
        np_array = np_array.astype(np.int32)
        torch_array = torch.from_numpy(np_array)
    elif dtype == np.uint8:
        torch_array = torch.frombuffer(
            pyvips_image.write_to_memory(), dtype=torch.uint8
            ).reshape(pyvips_image.height, pyvips_image.width, pyvips_image.bands)
    else:
        raise NotImplementedError
    torch_array = torch.transpose(torch_array, 0, 2)
    return torch_array


def calculate_magnification(mpp: float) -> Tuple[float, Optional[int]]:
    """
    Calculate the magnification of a microscope slide from its mpp (microns per pixel).

    Args:
    mpp (float): The microns per pixel value of the slide image.

    Returns:
    float: The estimated magnification of the slide.
    """
    magnifications = np.array([80, 40, 20, 10, 5])
    absolute_magnification = (0.25 / mpp) * 40
    if absolute_magnification < 2.5:
        magnification = None
    else:
        dists = np.abs(absolute_magnification - magnifications)
        mag_min_idx = dists.argmin()
        magnification = magnifications[mag_min_idx]
    return absolute_magnification, magnification
