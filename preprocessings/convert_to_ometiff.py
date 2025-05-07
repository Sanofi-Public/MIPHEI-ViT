import pyvips
import pandas as pd
from slidevips import SlideVips
from slidevips.ome_metadata import adapt_ome_metadata
from pathlib import Path


# CHANNEL_NAMES = ["Dapi", "CD4", "Pan-CK", "FoxP3", "CD56", "CD8", "CD3", "AF"] # IF3


def convert_to_ometiff(slide_path, channel_names, output_dir): # if_slide_path

    slide_name = Path(slide_path).stem
    slide = SlideVips(slide_path, mode="IF")
    resolution = slide.mpp
    magnification = slide.magnification
    n_channels = slide.n_channels
    slide.close()


    output_path = str(Path(output_dir) / (slide_name + ".ome.tiff"))
    pyvips_image = pyvips.Image.new_from_file(slide_path, n=n_channels, access="sequential")
    pyvips_image = pyvips_image.copy()

    ome_xml_metadata = adapt_ome_metadata(pyvips_image, resolution, channel_names, magnification)
    image_height = pyvips_image.height // n_channels
    pyvips_image.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    pyvips_image.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)
    pyvips_image.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / resolution,
        yres=1000 / resolution,
        page_height=image_height)
