"""
PathoCell preprocessing pipeline

Steps:
1. Build cell-level dataframe (fine + coarse gt, one-hot, parquet)
2. Compute global IF foreground/background percentiles (per channel)
3. Write IF OME-TIFF WSIs (normalized + log-compressed)
4. Write H&E and nuclei OME-TIFF WSIs
5. Build slide dataframe and tile dataframe (for tiling)
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import h5py
import cv2
from tqdm import tqdm
from skimage.filters import threshold_triangle
import pyvips

from slidevips.ome_metadata import adapt_ome_metadata_he, adapt_ome_metadata
from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu



PATHOCELL_CHANNEL_NAMES = [
    "CD44", "FOXP3", "CDX2", "CD8a", "p53", "GATA3", "CD45", "T-bet",
    "beta-catenin", "HLA-DR", "PD-L1", "Ki67", "CD45RA", "CD4",
    "CD21", "MUC-1", "CD30", "CD2", "Vimentin", "CD20", "LAG-3", "Na-K-ATPase",
    "CD5", "IDO-1", "Cytokeratin", "CD11b", "CD56", "SMA", "BCL-2", "CD25",
    "Collagen IV", "CD11c", "PD-1", "HOCHST13", "Granzyme B", "EGFR", "VISTA", "CD15",
    "CD194", "ICOS", "MMP9", "Synaptophysin", "CD71", "GFAP", "CD7", "CD3e",
    "Chromogranin A", "CD163", "CD57", "CD45RO", "CD68", "CD31", "Podoplanin",
    "CD34", "CD38", "CD138", "MMP12", "DRAQ5",
]

# =========================================================
# OME-TIFF WRITERS
# =========================================================

def write_if_ome_tiff(image: np.ndarray, output_path: str, slide_mpp: float = 0.5) -> None:
    """
    Write IF channels as multi-channel OME-TIFF.
    image: H x W x C, uint8
    """
    image_pyvips = pyvips.Image.new_from_array(image).copy(interpretation="b-w").cast("uchar")
    num_channels = image_pyvips.bands

    # stack channels horizontally (as in original code)
    image_pyvips = (
        pyvips.Image.arrayjoin(image_pyvips.bandsplit(), across=1)
        .copy(interpretation="b-w")
    )

    magnification = int(10 / slide_mpp)

    assert len(PATHOCELL_CHANNEL_NAMES) == num_channels, "Channel name list does not match image bands."

    ome_xml_metadata = adapt_ome_metadata(image_pyvips, slide_mpp, PATHOCELL_CHANNEL_NAMES, magnification)

    # Each channel is stacked vertically, so compute the height of a single channel image
    # This step is necessary for OME-TIFF metadata. QuPath can then reconstruct the image correctly.
    image_height = image_pyvips.height / num_channels

    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    image_pyvips.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / slide_mpp,
        yres=1000 / slide_mpp,
        page_height=image_height,
    )


def write_nuclei_ome_tiff(image: np.ndarray, output_path: str, slide_mpp: float = 0.5) -> None:
    """
    Write nuclei instance map as single-channel OME-TIFF.
    image: H x W, uint16
    """
    image_pyvips = pyvips.Image.new_from_array(image).copy(interpretation="b-w").cast("ushort")

    magnification = int(10 / slide_mpp)
    ome_xml_metadata = adapt_ome_metadata(image_pyvips, slide_mpp, ["nuclei"], magnification)

    image_height = image_pyvips.height

    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    image_pyvips.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        region_shrink="nearest",
        bigtiff=True,
        subifd=True,
        xres=1000 / slide_mpp,
        yres=1000 / slide_mpp,
        page_height=image_height,
    )


def write_he_ome_tiff(image: np.ndarray, output_path: str, slide_mpp: float = 0.5) -> None:
    """
    Write H&E RGB image as OME-TIFF.
    image: H x W x 3, uint8
    """
    image_pyvips = pyvips.Image.new_from_array(image).copy(interpretation="srgb")

    magnification = int(10 / slide_mpp)
    ome_xml_metadata = adapt_ome_metadata_he(image_pyvips, slide_mpp, magnification)

    image_height = image_pyvips.height

    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    image_pyvips.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / slide_mpp,
        yres=1000 / slide_mpp,
        page_height=image_height,
    )


# =========================================================
# CELL-LEVEL DATAFRAME
# =========================================================

def build_cell_dataframe(hdf_paths: list[Path]) -> pd.DataFrame:
    """
    For each HDF5, compute per-nucleus fine and coarse class (majority vote),
    then build a cell-level dataframe.
    """
    records = []

    for hdf_path in tqdm(hdf_paths, desc="Building cell-level table"):
        with h5py.File(hdf_path, "r") as f:
            gt_ct = f["gt_ct"][0, :].astype(np.int32)
            gt_ct_coarse = f["gt_ct_coarse"][0, :].astype(np.int32)
            image_inst = f["gt_inst"][0, :].astype(np.int32)

        labels = np.unique(image_inst)
        labels = labels[labels != 0]

        for lbl in labels:
            mask = image_inst == lbl

            vals_ct, counts_ct = np.unique(gt_ct[mask], return_counts=True)
            major_idx_ct = counts_ct.argmax()
            major_class_ct = int(vals_ct[major_idx_ct])

            vals_coarse, counts_coarse = np.unique(gt_ct_coarse[mask], return_counts=True)
            major_idx_coarse = counts_coarse.argmax()
            major_class_coarse = int(vals_coarse[major_idx_coarse])

            records.append({
                "in_slide_name": hdf_path.stem,
                "cell_id": int(lbl),
                "gt_ct": major_class_ct,
                "gt_ct_coarse": major_class_coarse,
            })

    df = pd.DataFrame(records)
    return df


def encode_cell_dataframe(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode coarse cell types and prepare for parquet export.
    """
    nuclei_classes = [
        'Background', 'B cells', 'Macrophages/Monocytes', 'Adipocytes',
        'Dendritic cells', 'T cells', 'Granulocytes', 'NK cells', 'Nerves',
        'Plasma cells', 'Smooth muscle', 'Stroma', 'Tumor cells',
        'Vasculature/Lymphatics', 'Other cells'
    ]

    cell_df = cell_df.rename(columns={"in_slide_name": "slide_name", "cell_id": "label"})
    # OME files are named <stem>.ome.tiff
    cell_df["slide_name"] = cell_df["slide_name"].apply(lambda x: x + ".ome")
    cell_df["slide_name"] = cell_df["slide_name"].astype("category")

    min_cell_class = cell_df["gt_ct_coarse"].min()
    cell_df["ct_name"] = cell_df["gt_ct_coarse"].map(
        lambda x: nuclei_classes[x - min_cell_class]
    )

    one_hot = pd.get_dummies(cell_df["ct_name"])
    cell_df["ct_name"] = cell_df["ct_name"].astype("category")
    cell_df = pd.concat([cell_df, one_hot], axis=1)

    # Optional: reorder columns a bit
    cols_first = ["slide_name", "label", "gt_ct", "gt_ct_coarse", "ct_name"]
    other_cols = [c for c in cell_df.columns if c not in cols_first]
    cell_df = cell_df[cols_first + other_cols]

    return cell_df


# =========================================================
# GLOBAL IF HISTOGRAMS & NORMALIZATION
# =========================================================

def infer_num_channels(hdf_paths: list[Path]) -> int:
    with h5py.File(hdf_paths[0], "r") as f:
        image_if = f["ifl"][:]  # (C, H, W)
    return image_if.shape[0]


def compute_if_percentiles(hdf_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel foreground / background percentiles (99.9% FG, 0.5% BG)
    based on triangle thresholding.
    """
    N_CHANNELS = infer_num_channels(hdf_paths)
    NBINS = 65536
    BIN_EDGES = np.linspace(0, 65536, NBINS + 1, dtype=np.float64)

    hist_fg = np.zeros((N_CHANNELS, NBINS), dtype=np.float64)
    hist_bg = np.zeros((N_CHANNELS, NBINS), dtype=np.float64)

    for hdf_path in tqdm(hdf_paths, desc="Accumulating IF histograms"):
        with h5py.File(hdf_path, "r") as f:
            image_if = f["ifl"][:]  # (C, H, W)

        for i, image_if_c in enumerate(image_if):
            thresh_c = threshold_triangle(image_if_c, nbins=NBINS)
            tissue_mask_c = image_if_c > thresh_c
            background_mask_c = ~tissue_mask_c

            h_fg, _ = np.histogram(image_if_c[tissue_mask_c], bins=BIN_EDGES)
            h_bg, _ = np.histogram(image_if_c[background_mask_c], bins=BIN_EDGES)

            hist_fg[i] += h_fg.astype(np.float64)
            hist_bg[i] += h_bg.astype(np.float64)

    cdf_fg = np.cumsum(hist_fg, axis=1)
    cdf_bg = np.cumsum(hist_bg, axis=1)

    cdf_fg /= cdf_fg[:, [-1]]
    cdf_bg /= cdf_bg[:, [-1]]

    def percentile_from_cdf(cdf_channel, percent):
        idx = np.searchsorted(cdf_channel, percent / 100.0)
        return BIN_EDGES[min(idx, len(BIN_EDGES) - 1)]

    p99_fg = np.asarray([percentile_from_cdf(cdf_fg_ch, 99.9) for cdf_fg_ch in cdf_fg])
    p05_bg = np.asarray([percentile_from_cdf(cdf_bg_ch, 0.5) for cdf_bg_ch in cdf_bg])

    return p99_fg, p05_bg


def write_if_omes(hdf_paths: list[Path], if_wsi_dir: Path,
                  p99_fg: np.ndarray, p05_bg: np.ndarray) -> None:
    if_wsi_dir.mkdir(exist_ok=True, parents=True)

    min_values = np.float32(p05_bg.reshape((1, 1, -1)))
    max_values = np.float32(p99_fg.reshape((1, 1, -1)))

    for hdf_path in tqdm(hdf_paths, desc="Writing IF OME-TIFFs"):
        with h5py.File(hdf_path, "r") as f:
            image_if = f["ifl"][:]  # (C, H, W)

        # transpose to (H, W, C)
        image_if = np.transpose(image_if, (1, 2, 0))

        image_if_clean = np.clip((image_if - min_values) / (max_values - min_values), 0.0, 1.0)
        image_if_clean = np.uint8(np.log(image_if_clean + 1) * 255)

        image_id = hdf_path.stem
        output_path_if = str(if_wsi_dir / f"{image_id}.ome.tiff")
        write_if_ome_tiff(image_if_clean, output_path_if)


# =========================================================
# H&E + NUCLEI OME-TIFFS
# =========================================================

def write_he_and_nuclei_omes(hdf_paths: list[Path],
                             he_wsi_dir: Path,
                             nuclei_wsi_dir: Path) -> None:
    he_wsi_dir.mkdir(exist_ok=True, parents=True)
    nuclei_wsi_dir.mkdir(exist_ok=True, parents=True)

    for hdf_path in tqdm(hdf_paths, desc="Writing H&E and nuclei OME-TIFFs"):
        with h5py.File(hdf_path, "r") as f:
            image_he = f["img"][:]      # (C, H, W)
            image_inst = f["gt_inst"][0, :]

        image_he = np.transpose(image_he, (1, 2, 0))  # (H, W, C)

        # tissue mask via Otsu on std
        _, tissue_mask = cv2.threshold(
            np.uint16(image_he.std(axis=-1)), 0, 65536,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        background_mask = tissue_mask == 0
        max_values = np.percentile(image_he[background_mask], 99, axis=0)

        image_he = np.uint8(np.clip(image_he / max_values, 0.0, 1.0) * 255)

        image_id = hdf_path.stem
        output_path_he = str(he_wsi_dir / f"{image_id}.ome.tiff")
        output_path_nuclei = str(nuclei_wsi_dir / f"{image_id}.ome.tiff")

        write_he_ome_tiff(image_he, output_path_he, slide_mpp=0.5)
        write_nuclei_ome_tiff(image_inst, output_path_nuclei, slide_mpp=0.5)


# =========================================================
# SLIDE + TILE DATAFRAMES
# =========================================================

def build_slide_and_tile_dataframes(
    data_dir: Path,
    he_wsi_dir: Path,
    if_wsi_dir: Path,
    nuclei_wsi_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    he_slide_paths = sorted(str(fn) for fn in he_wsi_dir.glob("*.ome.tiff"))

    slide_df = pd.DataFrame()
    slide_df["in_slide_path"] = he_slide_paths
    slide_df["in_slide_name"] = slide_df["in_slide_path"].apply(lambda x: Path(x).stem)
    slide_df["targ_slide_path"] = slide_df["in_slide_path"].apply(
        lambda x: str(if_wsi_dir / Path(x).name)
    )
    slide_df["nuclei_slide_path"] = slide_df["in_slide_path"].apply(
        lambda x: str(nuclei_wsi_dir / Path(x).name)
    )

    tile_rows = []
    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="Generating tiles"):
        slide_nuclei = SlideVips(row["nuclei_slide_path"], mode="IF")
        thumbnail = slide_nuclei.read_region((0, 0), 0, slide_nuclei.dimensions) > 0
        tile_positions, _ = get_locs_otsu(thumbnail, slide_nuclei.dimensions, 256)
        slide_nuclei.close()

        df_tile = pd.DataFrame({
            "in_slide_name": row["in_slide_name"],
            "x": tile_positions[:, 0],
            "y": tile_positions[:, 1],
            "level": 0,
            "tile_size_x": 256,
            "tile_size_y": 256,
        })
        tile_rows.append(df_tile)

    tile_df = pd.concat(tile_rows, ignore_index=True)
    return slide_df, tile_df


# =========================================================
# MAIN
# =========================================================

def main(data_dir: Path):
    data_dir = data_dir.resolve()
    print(f"📁 Using PathoCell data directory: {data_dir}")

    hdf_dir = data_dir / "pathocell_hdf"
    hdf_paths = sorted(p for p in hdf_dir.glob("*.hdf"))
    print(f"Found {len(hdf_paths)} HDF files.")

    # ---------- Step 1: Cell-level dataframe ----------
    cell_raw_df = build_cell_dataframe(hdf_paths)

    cell_df = encode_cell_dataframe(cell_raw_df)
    cell_parquet = data_dir / "cell_dataframe.parquet"
    cell_df.to_parquet(cell_parquet, compression=None)
    print(f"Saved encoded cell table → {cell_parquet}")

    # ---------- Step 2: IF global hist percentiles ----------
    p99_fg, p05_bg = compute_if_percentiles(hdf_paths)
    print("Computed IF percentiles (foreground 99.9%, background 0.5%).")

    # ---------- Step 3: Write IF OME-TIFFs ----------
    if_wsi_dir = data_dir / "omes" / "if"
    write_if_omes(hdf_paths, if_wsi_dir, p99_fg, p05_bg)

    # ---------- Step 4: Write H&E and nuclei OME-TIFFs ----------
    he_wsi_dir = data_dir / "omes" / "he"
    nuclei_wsi_dir = data_dir / "omes" / "nuclei"
    write_he_and_nuclei_omes(hdf_paths, he_wsi_dir, nuclei_wsi_dir)

    # ---------- Step 5: Slide + tile dataframes ----------
    slide_df, tile_df = build_slide_and_tile_dataframes(
        data_dir, he_wsi_dir, if_wsi_dir, nuclei_wsi_dir
    )
    assert len(slide_df) == 109
    slide_csv = data_dir / "slide_dataframe.csv"
    tile_csv = data_dir / "dataframe.csv"
    slide_df.to_csv(slide_csv, index=False)
    tile_df.to_csv(tile_csv, index=False)
    print(f"Saved slide dataframe   → {slide_csv}")
    print(f"Saved tile dataframe    → {tile_csv}")

    print("✅ PathoCell preprocessing completed.")

    # ---------- Step 6: marker_metadata.csv ----------

    marker_df = pd.DataFrame(columns=["Marker Name", "Index"])
    marker_df["Marker Name"] = PATHOCELL_CHANNEL_NAMES
    marker_df["Index"] = np.arange(len(PATHOCELL_CHANNEL_NAMES))
    marker_csv = data_dir / "marker_metadata.csv"
    marker_df.to_csv(marker_csv, index=False)
    print(f"Saved marker metadata → {marker_csv}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathoCell preprocessing.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to PathoCell root folder (must contain 'pathocell_hdf/').",
    )
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir) / "pathocell")
    main(Path(args.data_dir))
