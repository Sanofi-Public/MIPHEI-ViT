"""Adapt OME metadata for a given image."""

from pathlib import Path
from typing import List

import ome_types
import pyvips


current_script_path = Path(__file__)
xml_file_path = str(current_script_path.parent.parent / "default_pred_ome_config.xml")
xml_file_path_he = str(current_script_path.parent.parent / "default_he_ome_config.xml")

DEFAUT_CONFIG = ome_types.from_xml(xml_file_path)
DEFAUT_CONFIG_HE = ome_types.from_xml(xml_file_path_he)


PYVIPS2OME_FORMAT = {
    pyvips.BandFormat.UCHAR: "uint8",  # Unsigned 8-bit integer
    pyvips.BandFormat.CHAR: "int8",  # Signed 8-bit integer
    pyvips.BandFormat.USHORT: "uint16",  # Unsigned 16-bit integer
    pyvips.BandFormat.SHORT: "int16",  # Signed 16-bit integer
    pyvips.BandFormat.UINT: "uint32",  # Unsigned 32-bit integer
    pyvips.BandFormat.INT: "int32",  # Signed 32-bit integer
    pyvips.BandFormat.FLOAT: "float",  # 32-bit floating point
    pyvips.BandFormat.DOUBLE: "double",  # 64-bit floating point
    pyvips.BandFormat.COMPLEX: "complex64",  # Complex number, 2x 32-bit floating point
    pyvips.BandFormat.DPCOMPLEX: "complex128",  # Complex number, 2x 64-bit floating point
}


def adapt_ome_metadata(pyvips_image: pyvips.Image, resolution: float,
                       channel_names: List[str], magnification: float):
    """
    Adapt and generate OME-XML metadata for a given mIF pyvips image.

    This function uses default_pred_ome_config default config and overwrite it, to create
    OME-XML metadata for mIF image. Used when saving a mIF image in OME.TIFF.
    Args:
        pyvips_image: The pyvips mIF image containing the final image data.
        resolution (float): The physical size per pixel (in microns) for both X and Y axes.
        channel_names (List[str]): List of channel names corresponding to the image channels.
        magnification (float): The nominal magnification value for the objective.
    Returns:
        str: The generated OME-XML metadata as a string.
    Raises:
        AssertionError: If the image height is not divisible by the number of channels.
    """
    xml_config = DEFAUT_CONFIG
    n_channels = len(channel_names)

    ome_format = PYVIPS2OME_FORMAT[pyvips_image.format]
    assert pyvips_image.height % n_channels == 0

    xml_config.images[0].pixels.size_c = n_channels
    xml_config.images[0].pixels.type = ome_format
    xml_config.images[0].pixels.size_x = pyvips_image.width
    xml_config.images[0].pixels.size_y = pyvips_image.height // n_channels
    xml_config.images[0].pixels.physical_size_x = resolution
    xml_config.images[0].pixels.physical_size_y = resolution

    planes = [ome_types.model.Plane(the_z=0, the_t=0, the_c=idx_c) for idx_c in range(n_channels)]
    channels = [ome_types.model.Channel(
        id=f'Channel:{idx_c}', name=channel_names[idx_c], samples_per_pixel=1,
        light_path={}) for idx_c in range(n_channels)]
    xml_config.images[0].pixels.planes = planes
    xml_config.images[0].pixels.channels = channels
    xml_config.instruments[0].objectives[0].nominal_magnification = magnification
    return xml_config.to_xml()


def adapt_ome_metadata_he(final_img, resolution, magnification):
    """
    Adapt and generate OME-XML metadata for a given H&E pyvips image.

    This function uses default_he_ome_config default config and overwrite it, to create
    OME-XML metadata for H&E image. Used when saving a H&E image in OME.TIFF.
    Args:
        pyvips_image: The pyvips H&E image containing the final image data.
        resolution (float): The physical size per pixel (in microns) for both X and Y axes.
        channel_names (List[str]): List of channel names corresponding to the image channels.
        magnification (float): The nominal magnification value for the objective.
    Returns:
        str: The generated OME-XML metadata as a string.
    Raises:
        AssertionError: If the image height is not divisible by the number of channels.
    """
    xml_config = DEFAUT_CONFIG_HE  # already loaded from XML

    pixels = xml_config.images[0].pixels
    pixels.size_x = final_img.width
    pixels.size_y = final_img.height
    pixels.physical_size_x = resolution
    pixels.physical_size_y = resolution
    pixels.physical_size_x_unit = "µm"
    pixels.physical_size_y_unit = "µm"

    if xml_config.instruments and xml_config.instruments[0].objectives:
        xml_config.instruments[0].objectives[0].nominal_magnification = magnification

    return xml_config.to_xml()
