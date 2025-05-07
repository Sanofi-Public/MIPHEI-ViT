from collections import defaultdict
from pathlib import Path
import pyvips
from typing import List, Optional, Tuple
import numpy as np
import ome_types


def get_pyramid_pyvips(filename: str, channel_idxs: Optional[List[int]] = None,
                       mode: Optional[str] = None) -> Tuple[List[pyvips.Image], dict]:
    """
    Retrieves a pyramid of images from a given filename using pyvips library and the
        corresponding metadata.

    Args:
        filename (str): The path to the image file.
        channel_idxs (list, optional): List of channel indexes to extract from the image.
            Defaults to None.
        mode (str, optional): The mode of the image. Defaults to None.

    Returns:
        tuple: A tuple containing the pyramid of images and the image fields.
    """
    pyramid_image = []
    image_format = Path(filename).suffix
    image = pyvips.Image.new_from_file(filename, access="sequential")
    fields = {field: image.get(field) for field in image.get_fields()}
    if mode == "RGB" and image.bands == 4:
        channel_idxs = [0, 1, 2]

    if image_format in [".ndpi", ".svs"]:
        mppx = float(fields["openslide.mpp-x"])
        mppy = float(fields["openslide.mpp-y"])
        n_levels = int(image.get("openslide.level-count"))
        for level in range(n_levels):
            image = pyvips.Image.new_from_file(filename, level=level, access="sequential")
            if channel_idxs is not None:
                image = image[channel_idxs]
            pyramid_image.append(image)

    elif image_format in [".tif", ".tiff", ".ome.tiff", ".ome.tif"]:
        pixels_metadata = ome_types.from_xml(
            fields['image-description']).images[0].pixels
        mppx = pixels_metadata.physical_size_x
        mppy = pixels_metadata.physical_size_x
        n_pages, n_subifds = image.get("n-pages"), image.get("n-subifds")
        del image
        for level in range(-1, n_subifds):
            if channel_idxs is None:
                channels = [pyvips.Image.new_from_file(
                    filename, subifd=level, page=channel,
                    access="sequential") for channel in range(n_pages)]
            else:
                channels = [pyvips.Image.new_from_file(
                    filename, subifd=level, page=channel,
                    access="sequential") for channel in channel_idxs]
            image = channels[0].bandjoin(channels[1:])
            pyramid_image.append(image)
    elif image_format == ".qptiff":
        mppx = 1 / fields["xres"]
        mppy = 1 / fields["yres"]
        if fields["resolution-unit"] == "cm":
            mppx *= 1000
            mppy *= 1000
        else:
            raise ValueError("Unknown resolution unit")
        n_pages = image.get("n-pages")
        del image

        area2channels = defaultdict(list)
        for page in range(n_pages):
            channel_level = pyvips.Image.new_from_file(filename, page=page, access="sequential")
            if channel_level.bands > 1:
                del channel_level
                continue
            area = channel_level.width * channel_level.height
            area2channels[area].append(channel_level)

        pyramid_image = []
        areas = list(area2channels.keys())
        nb_bands = len(area2channels[areas[0]])
        for area in sorted(areas, reverse=True):
            channels = area2channels[area]
            assert len(channels) == nb_bands
            if channel_idxs is not None:
                channels = [channels[channel_idx] for channel_idx in channel_idxs]
            image = channels[0].bandjoin(channels[1:])
            pyramid_image.append(image)
    elif image_format == "qptiff":
        raise NotImplementedError

    else:
        raise NotImplementedError

    del image
    assert np.abs(mppx - mppy < 1e-3)
    mpp = (mppx + mppy) / 2
    assert 0 < mpp < 15
    fields["mpp"] = mpp
    return pyramid_image, fields
