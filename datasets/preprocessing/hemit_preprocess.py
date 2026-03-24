"""
HEMIT preprocessing:
- Build slide + train/val/test dataframes (image, target, nuclei mask, nuclei CSV)
- Extract nuclei-level cell table from per-FOV CSVs
- Save nuclei dataframe as Parquet
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm


HEMIT_CHANNEL_NAMES = [
    "Pan-CK",
    "CD3",
    "Dapi",
]


# =========================================================
# STEP 1 — BUILD DATAFRAMES (train/val/test/slide)
# =========================================================

def build_hemit_dataframes(data_dir: Path, output_dir: Path) -> None:
    """
    Generate and save dataframes containing image paths and related metadata
    for the HEMIT dataset.

    This scans:
        <data_dir>/train/input/*.tif
        <data_dir>/val/input/*.tif
        <data_dir>/test/input/*.tif

    and builds corresponding target, nuclei mask, and nuclei CSV paths.

    Saves:
        - slide_dataframe_hemit.csv
        - dataframe_hemit.csv
        - train_dataframe_hemit.csv
        - val_dataframe_hemit.csv
        - test_dataframe_hemit.csv
    """
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = ["in_slide_name", "image_path", "target_path", "nuclei_path", "nuclei_csv_path"]
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)

    # ----------------- TRAIN -----------------
    train_df["image_path"] = [str(fn) for fn in (data_dir / "train" / "input").glob("*.tif")]
    train_df["in_slide_name"] = train_df["image_path"].apply(lambda x: Path(x).name)
    train_df["target_path"] = train_df["image_path"].apply(
        lambda x: x.replace("/input/", "/label/")
    )
    train_df["nuclei_path"] = train_df["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/")
    )
    train_df["nuclei_csv_path"] = train_df["image_path"].apply(
        lambda x: x.replace("/input/", "/csv/").replace(".tif", ".csv")
    )

    # ----------------- VAL -----------------
    val_df["image_path"] = [str(fn) for fn in (data_dir / "val" / "input").glob("*.tif")]
    val_df["in_slide_name"] = val_df["image_path"].apply(lambda x: Path(x).name)
    val_df["target_path"] = val_df["image_path"].apply(
        lambda x: x.replace("/input/", "/label/")
    )
    val_df["nuclei_path"] = val_df["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/")
    )
    val_df["nuclei_csv_path"] = val_df["image_path"].apply(
        lambda x: x.replace("/input/", "/csv/").replace(".tif", ".csv")
    )

    # ----------------- TEST -----------------
    test_df["image_path"] = [str(fn) for fn in (data_dir / "test" / "input").glob("*.tif")]
    test_df["in_slide_name"] = test_df["image_path"].apply(lambda x: Path(x).name)
    test_df["target_path"] = test_df["image_path"].apply(
        lambda x: x.replace("/input/", "/label/")
    )
    test_df["nuclei_path"] = test_df["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/")
    )
    test_df["nuclei_csv_path"] = test_df["image_path"].apply(
        lambda x: x.replace("/input/", "/csv/").replace(".tif", ".csv")
    )

    print(f"Found {len(train_df)} train, {len(val_df)} val, {len(test_df)} test tiles.")

    # ----------------- COMBINED DATAFRAME -----------------
    dataframe = pd.concat((train_df, val_df, test_df), ignore_index=True)

    # Basic existence checks
    assert dataframe["image_path"].apply(lambda x: Path(x).exists()).all(), "Missing H&E images"
    assert dataframe["target_path"].apply(lambda x: Path(x).exists()).all(), "Missing IF labels"
    assert dataframe["nuclei_path"].apply(lambda x: Path(x).exists()).all(), "Missing nuclei masks"

    # ----------------- SLIDE DATAFRAME -----------------
    slide_df = pd.DataFrame()
    slide_df["in_slide_name"] = dataframe["in_slide_name"]
    slide_df["nuclei_csv_path"] = dataframe["nuclei_csv_path"]

    # NOTE: Some FOVs might be duplicated across splits;
    # we keep unique rows in slide dataframe.
    slide_df = slide_df.drop_duplicates(subset=["in_slide_name", "nuclei_csv_path"])

    # ----------------- SAVE -----------------
    (output_dir / "slide_dataframe_hemit.csv").write_text(
        slide_df.to_csv(index=False)
    )
    (output_dir / "dataframe_hemit.csv").write_text(
        dataframe.to_csv(index=False)
    )
    (output_dir / "train_dataframe_hemit.csv").write_text(
        train_df.to_csv(index=False)
    )
    (output_dir / "val_dataframe_hemit.csv").write_text(
        val_df.to_csv(index=False)
    )
    (output_dir / "test_dataframe_hemit.csv").write_text(
        test_df.to_csv(index=False)
    )

    print(f"Saved slide/dataframes in: {output_dir}")
    print(" - slide_dataframe_hemit.csv")
    print(" - dataframe_hemit.csv")
    print(" - train_dataframe_hemit.csv")
    print(" - val_dataframe_hemit.csv")
    print(" - test_dataframe_hemit.csv")
    

# =========================================================
# STEP 2 — BUILD NUCLEI PARQUET FROM CSVs
# =========================================================

DROP_COLUMNS = ["area", "X_centroid", "Y_centroid", "Pan-CK", "CD3", "DAPI"]


def build_nuclei_dataframe_from_csvs(
    slide_dataframe_path: Path,
    output_parquet_path: Path,
) -> None:
    """
    Read all nuclei CSV files referenced in slide_dataframe_hemit.csv,
    drop unwanted columns, attach slide_name, and save a combined Parquet.

    Assumes slide_dataframe_hemit.csv has columns:
        - in_slide_name
        - nuclei_csv_path
    """
    slide_df = pd.read_csv(slide_dataframe_path)
    csv_paths: List[str] = slide_df["nuclei_csv_path"].unique().tolist()

    print(f"Found {len(csv_paths)} nuclei CSV files.")

    nuclei_rows = []
    for csv_path in tqdm(csv_paths, desc="Building nuclei dataframe"):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue

        df_tile = pd.read_csv(csv_path)

        # Drop extra columns if present
        cols_to_drop = [c for c in DROP_COLUMNS if c in df_tile.columns]
        if len(cols_to_drop) > 0:
            df_tile = df_tile.drop(columns=cols_to_drop)

        # slide_name: match original behaviour (stem + .tif)
        fov_name = csv_path.stem + ".tif"
        df_tile["slide_name"] = pd.Categorical([fov_name] * len(df_tile))

        nuclei_rows.append(df_tile)

    if not nuclei_rows:
        raise RuntimeError("No nuclei rows found; check your nuclei_csv_path values.")

    nuclei_df = pd.concat(nuclei_rows, ignore_index=True)
    nuclei_df["slide_name"] = nuclei_df["slide_name"].astype("category")

    output_parquet_path = output_parquet_path.resolve()
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei_df.to_parquet(output_parquet_path, compression=None)

    print(f"Saved nuclei dataframe Parquet → {output_parquet_path}")
    print(f"Total nuclei rows: {len(nuclei_df)}")


# =========================================================
# MAIN
# =========================================================

def main(data_dir: Path):
    output_dir = data_dir
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Using HEMIT data_dir:   {data_dir}")
    print(f"📁 Using output_dir:      {output_dir}")

    # Step 1: build train/val/test/slide dataframes
    build_hemit_dataframes(data_dir=data_dir, output_dir=output_dir)

    # Step 2: build nuclei_dataframe.parquet
    slide_df_path = output_dir / "slide_dataframe_hemit.csv"
    nuclei_parquet_path = output_dir / "nuclei_dataframe.parquet"
    build_nuclei_dataframe_from_csvs(slide_df_path, nuclei_parquet_path)

    # Step 3 — marker_metadata

    marker_df = pd.DataFrame(columns=["Marker Name", "Index"])
    marker_df["Marker Name"] = HEMIT_CHANNEL_NAMES
    marker_df["Index"] = np.arange(len(HEMIT_CHANNEL_NAMES))
    marker_csv = data_dir / "marker_metadata.csv"
    marker_df.to_csv(marker_csv, index=False)
    print(f"Saved marker metadata → {marker_csv}")
    print("✨ HEMIT preprocessing completed.")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEMIT preprocessing pipeline.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the HEMIT dataset root (containing train/val/test folders).",
    )
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir) / "HEMIT_dataset")
    main(Path(args.data_dir))
