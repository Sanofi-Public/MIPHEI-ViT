"""
Script to generate tile-level dataframes from H&E ORION WSI for training, validation, and testing.

Scans a directory of slide folders, extracts tile positions using Otsu thresholding, and saves
metadata as CSV files.
"""

import argparse

from pathlib import Path
import pandas as pd

from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu


TEST_SLIDES = ['19510_C11_US_SCAN_OR_001__151039-registered.ome',
               '18459_LSP10364_US_SCAN_OR_001__092347-registered.ome']
VAL_SLIDES = ['19510_C19_US_SCAN_OR_001__153041-registered.ome',
              '19510_C30_US_SCAN_OR_001__155702-registered.ome']


def main(data_dir: Path, dataframe_folder_save: Path, tile_size: int = 512,
         tile_overlap: int = 0, level: int = 0) -> None:
    """
    Process H&E ORION WSI to create tile-level dataframes for training, validation, and testing.

    This function scans the provided ORION data directory for slide image folders, extracts
    relevant file paths, computes tile positions using Otsu thresholding, and generates CSV
    dataframes containing tile metadata. The data is split into training, validation, and test sets
    based on predefined slide names.
    Args:
        data_dir (Path): Path to the directory containing slide subdirectories.
        tile_size (int): Size of the square tile to extract from slides.
        tile_overlap (int): Overlap between adjacent tiles in pixels.
        level (int): Pyramid level of the slide to process.
        dataframe_folder_save (Path): Directory where the resulting dataframes will be saved.
    Returns:
        None
    """
    dirs_paths = list(data_dir.glob("*"))
    rows = []
    for dirs_path in dirs_paths:
        he_path = str(list(dirs_path.glob("*registered.ome.tif"))[0])
        he_name = Path(he_path).stem
        if_path = str(list(dirs_path.glob("*zlib.ome.tiff"))[0])
        assert he_path
        assert if_path
        rows.append([he_name, he_path, if_path])
    print(len(rows))

    slide_dataframe = pd.DataFrame(rows,
                                   columns=["in_slide_name", "in_slide_path", "targ_slide_path"])
    slide_dataframe.to_csv(str(dataframe_folder_save / "slide_dataframe.csv"), index=False)

    slide_names = []
    xs = []
    ys = []

    for _, row in slide_dataframe.iterrows():
        slide_name, he_path = row["in_slide_name"], row["in_slide_path"]
        slide_he = SlideVips(he_path)

        thumbnail = slide_he.get_thumbnail((3000, 3000))
        tile_positions, _ = get_locs_otsu(
            thumbnail, slide_he.level_dimensions[level], tile_size, tile_overlap, 0.07)

        slide_he.close()
        slide_names += [slide_name] * len(tile_positions)
        xs += tile_positions[..., 0].tolist()
        ys += tile_positions[..., 1].tolist()

    dataframe = pd.DataFrame(
        columns=["in_slide_name", "x", "y", "level", "tile_size_x", "tile_size_y"])
    dataframe["in_slide_name"] = slide_names
    dataframe["x"] = xs
    dataframe["y"] = ys
    dataframe["level"] = level
    dataframe["tile_size_x"] = tile_size
    dataframe["tile_size_y"] = tile_size

    dataframe.to_csv(str(dataframe_folder_save / "dataframe.csv"), index=False)

    train_dataframe = dataframe[~dataframe["in_slide_name"].isin(VAL_SLIDES + TEST_SLIDES)]
    val_dataframe = dataframe[dataframe["in_slide_name"].isin(VAL_SLIDES)]
    test_dataframe = dataframe[dataframe["in_slide_name"].isin(TEST_SLIDES)]

    print(len(train_dataframe), len(val_dataframe), len(test_dataframe))
    train_dataframe.to_csv(str(dataframe_folder_save / "train_dataframe.csv"), index=False)
    val_dataframe.to_csv(str(dataframe_folder_save / "val_dataframe.csv"), index=False)
    test_dataframe.to_csv(str(dataframe_folder_save / "test_dataframe.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tile-level dataframe from slide images.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--dataframe_folder_save", type=Path, required=True,
                        help="Folder to save output dataframes")

    parser.add_argument("--tile_size", type=int, default=512, help="Tile size (default: 512)")
    parser.add_argument("--tile_overlap", type=int, default=0, help="Tile overlap (default: 0)")
    parser.add_argument("--level", type=int, default=0, help="Pyramid level (default: 0)")

    args = parser.parse_args()

    main(
        args.data_dir,
        args.dataframe_folder_save,
        args.tile_size,
        args.tile_overlap,
        args.level
    )
