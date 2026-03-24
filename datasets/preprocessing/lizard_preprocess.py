"""
Lizard preprocessing:
- Build slide dataframe (image, inst_map, mat labels)
- Extract nuclei-level cell types (CSV + Parquet)
- Convert instance maps to TIFF
- Generate tile locations with SlideVips
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from PIL import Image

import pyvips
from slidevips import SlideVips
from slidevips.reader import NUMPY_DTYPE_MAPPING
from slidevips.tiling import get_locs_otsu
from typing import Optional, List


# =========================================================
# STEP 1 — SLIDE DATAFRAME
# =========================================================

def collect_image_paths(data_dir: Path) -> list[str]:
    """Collect all Lizard PNG images from both subsets."""
    p1 = data_dir / "lizard_images1" / "Lizard_Images1"
    p2 = data_dir / "lizard_images2" / "Lizard_Images2"

    paths = [str(fn) for fn in p1.glob("*.png")]
    paths += [str(fn) for fn in p2.glob("*.png")]
    return paths


def build_slide_dataframe(image_paths: list[str], data_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    df["in_slide_name"] = [Path(path).stem for path in image_paths]
    df["in_slide_path"] = image_paths

    df["nuclei_slide_path"] = df["in_slide_path"].apply(
        lambda x: str(data_dir / "inst_maps" / (Path(x).stem + ".tiff"))
    )
    df["label_slide_path"] = df["in_slide_path"].apply(
        lambda x: str(data_dir / "lizard_labels" / "Lizard_Labels" / "Labels" / (Path(x).stem + ".mat"))
    )

    return df


# =========================================================
# STEP 2 — NUCLEI CELL TYPES
# =========================================================

def extract_cell_table(slide_df: pd.DataFrame) -> pd.DataFrame:
    """Read all .mat label files and build nuclei cell-level table."""
    all_rows = []
    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="Extracting cell tables"):
        mat = sio.loadmat(row["label_slide_path"])
        cell_ids = mat["id"].flatten()
        cell_classes = mat["class"].flatten()

        df = pd.DataFrame({
            "in_slide_name": row["in_slide_name"],
            "cell_id": cell_ids,
            "cell_class": cell_classes
        })
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def one_hot_encode(cell_df: pd.DataFrame, nuclei_classes: list[str]) -> pd.DataFrame:
    cell_df = cell_df.rename(columns={"in_slide_name": "slide_name", "cell_id": "label"})
    cell_df["slide_name"] = cell_df["slide_name"].astype("category")

    min_class = cell_df["cell_class"].min()
    cell_df["ct_name"] = cell_df["cell_class"].map(lambda x: nuclei_classes[x - min_class])

    one_hot = pd.get_dummies(cell_df["ct_name"])
    cell_df = pd.concat([cell_df, one_hot], axis=1)
    cell_df["ct_name"] = cell_df["ct_name"].astype("category")
    return cell_df


# =========================================================
# STEP 3 — INSTANCE MAP TIFF GENERATION
# =========================================================

def convert_inst_maps(slide_df: pd.DataFrame, inst_dir: Path):
    inst_dir.mkdir(exist_ok=True)

    max_val = 0
    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="Checking max instance ID"):
        mat = sio.loadmat(row["label_slide_path"])
        inst_map = mat["inst_map"]
        max_val = max(max_val, inst_map.max())

    assert max_val < 2**16, f"Instance IDs exceed uint16 range: {max_val}"

    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="Saving TIFF instance maps"):
        mat = sio.loadmat(row["label_slide_path"])
        inst_map = np.uint16(mat["inst_map"])
        slide_name = row["in_slide_name"]
        Image.fromarray(inst_map).save(inst_dir / f"{slide_name}.tiff")


# =========================================================
# STEP 4 — TILE GENERATION USING SLIDEVIPS
# =========================================================

def get_png_pyramid(filepath: str, channel_idxs: Optional[List[int]] = None):
    arr = np.asarray(Image.open(filepath))
    im = pyvips.Image.new_from_array(arr)
    if channel_idxs is not None:
        im = im[channel_idxs]
    return [im]


class SlideVipsPNG(SlideVips):
    """Minimal SlideVips wrapper to read PNG nuclei maps."""
    def __init__(self, filepath: str, mpp: float, channel_idxs: List[int] = None):
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0)
        pyvips.cache_set_max_files(0)
        assert os.path.exists(filepath)

        self.pyramid_image = get_png_pyramid(filepath, channel_idxs)
        self.fields = {"mpp": mpp}
        self.level_count = 1
        self.mpp = mpp

        self.compute_spatial_attributes(self.mpp)

        self.slide_name = Path(filepath).stem
        self.n_channels = self.pyramid_image[0].bands
        self.dtype = self.pyramid_image[0].get("format")
        self.dtype_numpy = NUMPY_DTYPE_MAPPING[self.dtype]
        self._reiter_fetch = False


def build_tile_dataframe(slide_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="Generating tiles"):
        slide = SlideVipsPNG(row["nuclei_slide_path"], mpp=0.5)
        thumbnail = slide.read_region((0, 0), 0, slide.dimensions) > 0

        tile_positions, _ = get_locs_otsu(thumbnail, slide.dimensions, 256)

        df = pd.DataFrame({
            "in_slide_name": row["in_slide_name"],
            "x": tile_positions[:, 0],
            "y": tile_positions[:, 1],
            "level": 0,
            "tile_size_x": 256,
            "tile_size_y": 256,
        })
        records.append(df)

    return pd.concat(records, ignore_index=True)


# =========================================================
# MAIN
# =========================================================

def main(data_dir: Path):
    data_dir = data_dir.resolve()
    print(f"📁 Using Lizard directory: {data_dir}")

    # -------------------------
    # Step 1: Slide dataframe
    # -------------------------
    image_paths = collect_image_paths(data_dir)
    print(f"Found {len(image_paths)} images.")

    slide_df = build_slide_dataframe(image_paths, data_dir)
    slide_csv = data_dir / "slide_dataframe.csv"
    slide_df.to_csv(slide_csv, index=False)
    print(f"Saved slide dataframe → {slide_csv}")

    # -------------------------
    # Step 2: Cell-level table
    # -------------------------
    cell_df = extract_cell_table(slide_df)

    nuclei_classes = [
        "Neutrophil",
        "Epithelial",
        "Lymphocyte",
        "Plasma",
        "Eosinophil",
        "Connective_tissue"
    ]

    cell_df = one_hot_encode(cell_df, nuclei_classes)
    cell_parquet = data_dir / "nuclei_dataframe.parquet"
    cell_df.to_parquet(cell_parquet, compression=None)
    print(f"Saved Parquet → {cell_parquet}")

    # -------------------------
    # Step 3: Instance maps
    # -------------------------
    inst_dir = data_dir / "inst_maps"
    convert_inst_maps(slide_df, inst_dir)

    # -------------------------
    # Step 4: Tile dataframe
    # -------------------------
    tiles_df = build_tile_dataframe(slide_df)
    tiles_csv = data_dir / "dataframe.csv"
    tiles_df.to_csv(tiles_csv, index=False)
    print(f"Saved tile dataframe → {tiles_csv}")

    print("✨ Lizard preprocessing completed!")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lizard preprocessing pipeline.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to Lizard dataset root directory."
    )
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir) / "lizard")
    main(Path(args.data_dir))
