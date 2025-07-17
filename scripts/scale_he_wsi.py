"""Script to scale H&E WSIs to a target microns-per-pixel (MPP) resolution, saving them as \
OME-TIFF files with updated metadata."""

import argparse
from pathlib import Path

import pandas as pd
import pyvips
from slidevips import SlideVips
from tqdm import tqdm

import ome_types


def write_scaled_wsi(slide_dataframe: pd.DataFrame, out_dir: Path, mpp_target: float,
                     compression: str = "jpeg") -> None:
    """
    Scale and write H&E WSIs to a target microns-per-pixel (MPP) resolution, saving them as \
    OME-TIFF files with updated metadata.

    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing H&E input slide paths under the column
            "in_slide_path".
        out_dir (Path): Directory where the scaled WSIs will be saved.
        mpp_target (float): Target microns-per-pixel (MPP) value for scaling the slides.
        compression (str, optional): Compression type for TIFF saving (e.g., "jpeg").
            cf https://libvips.github.io/pyvips/enums.html#pyvips.enums.ForeignTiffCompression
            Defaults to "jpeg".
    Returns:
        None
    Raises:
        KeyError: If "in_slide_path" column is missing in the DataFrame.
        FileNotFoundError: If any input slide path does not exist.
        Exception: For errors during slide processing or saving.
    """
    paths = slide_dataframe["in_slide_path"].to_list()

    for path in tqdm(paths):
        output_path = str(out_dir / Path(path).name)
        slide = SlideVips(path)
        scale = slide.mpp / mpp_target
        slide.resize(scale)

        final_img = slide.pyramid_image[0]
        final_img = final_img.copy()
        image_height = final_img.height  # one channel only
        ome_metadata = ome_types.from_tiff(path)
        ome_metadata.images[0].pixels.type = "uint8"
        ome_metadata.images[0].pixels.size_x = final_img.width
        ome_metadata.images[0].pixels.size_y = image_height
        ome_metadata.images[0].pixels.physical_size_x = 0.245
        ome_metadata.images[0].pixels.physical_size_y = 0.245

        ome_xml_metadata = ome_metadata.to_xml()

        final_img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
        final_img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

        final_img.tiffsave(
            output_path,
            compression=compression,
            predictor="none",
            pyramid=True,
            tile=True,
            tile_width=512,
            tile_height=512,
            bigtiff=True,
            subifd=True,
            xres=1000 / 0.245,
            yres=1000 / 0.245,
            page_height=image_height)
        del final_img
        slide.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale H&E WSI slides to a target MPP.")
    parser.add_argument(
        "--slide_dataframe_path",
        type=str,
        required=True,
        help="Path to the slide dataframe CSV"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for scaled slides"
    )
    parser.add_argument(
        "--mpp_target",
        type=float,
        default=0.245,
        help="Target microns per pixel (default: 0.245)"
    )
    args = parser.parse_args()

    slide_dataframe = pd.read_csv(args.slide_dataframe_path)
    write_scaled_wsi(slide_dataframe, Path(args.out_dir), args.mpp_target)
