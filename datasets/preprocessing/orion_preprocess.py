"""
ORION preprocessing:
- Fix absolute paths in train/val/test dataframes
- Build slide dataframe (image_path, target_path, nuclei_path)
- Extract cell-level nuclei table from CSV files
- Save nuclei_dataframe.parquet
"""

import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import glob


ORION_CHANNEL_NAMES = [
    "Hoechst",
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "PD-1",
    "Ki67",
    "Pan-CK",
    "SMA",
]

# =========================================================
# STEP 1 — FIX DATAFRAME PATHS
# =========================================================

def load_and_fix_dataframe(csv_path: Path, data_dir: Path) -> pd.DataFrame:
    """
    Load ORION dataframe (train/val/test) and update:
      - image_path  -> data_dir / "he"     / filename
      - target_path -> data_dir / "if"     / filename
      - nuclei_path -> data_dir / "nuclei" / filename
    """
    df = pd.read_csv(csv_path)

    def fix_path(path: str, subfolder: str) -> str:
        return str(data_dir / subfolder / Path(path).name)

    df["image_path"]  = df["image_path"].apply(lambda p: fix_path(p, "he"))
    df["target_path"] = df["target_path"].apply(lambda p: fix_path(p, "if"))
    df["nuclei_path"] = df["nuclei_path"].apply(lambda p: fix_path(p, "nuclei"))

    return df


# =========================================================
# STEP 2 — BUILD SLIDE DATAFRAME
# =========================================================

def build_slide_dataframe(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    ORION slide-level dataframe only needs:
      - in_slide_name
      - nuclei_csv_path
    """
    df = pd.concat(df_list, ignore_index=True)
    slide_df = pd.DataFrame()
    slide_df["in_slide_name"] = df["in_slide_name"].astype("category")
    slide_df["nuclei_csv_path"] = df["nuclei_path"].apply(
        lambda p: p.replace(".tif", ".csv").replace("/nuclei/", "/csv_nuclei_pos/")
    )
    return slide_df



# =========================================================
# MAIN PIPELINE
# =========================================================

def main(data_dir: Path):
    data_dir = data_dir.resolve()
    print(f"📁 Using ORION directory: {data_dir}")

    # -----------------------------------------------------
    # Step 1 — Fix train/val/test dataframes
    # -----------------------------------------------------
    train_csv = data_dir / "train_dataframe.csv"
    val_csv   = data_dir / "val_dataframe.csv"
    test_csv  = data_dir / "test_dataframe.csv"

    assert train_csv.exists(), f"Missing {train_csv}"
    assert val_csv.exists(),   f"Missing {val_csv}"
    assert test_csv.exists(),  f"Missing {test_csv}"

    print("Fixing dataframe paths...")

    train_df = load_and_fix_dataframe(train_csv, data_dir)
    val_df   = load_and_fix_dataframe(val_csv, data_dir)
    test_df  = load_and_fix_dataframe(test_csv, data_dir)

    # Save fixed versions
    train_df.to_csv(data_dir / "train_dataframe.csv", index=False)
    val_df.to_csv(data_dir / "val_dataframe.csv", index=False)
    test_df.to_csv(data_dir / "test_dataframe.csv", index=False)

    print("Saved fixed dataframes.")

    # -----------------------------------------------------
    # Step 2 — Slide dataframe
    # -----------------------------------------------------
    slide_df = build_slide_dataframe([train_df, val_df, test_df])
    slide_df.to_csv(data_dir / "slide_dataframe.csv", index=False)
    print("Saved slide_dataframe.csv")

    # -----------------------------------------------------
    # Step 3 — marker_metadata
    # -----------------------------------------------------

    marker_df = pd.DataFrame(columns=["Marker Name", "Index"])
    marker_df["Marker Name"] = ORION_CHANNEL_NAMES
    marker_df["Index"] = np.arange(len(ORION_CHANNEL_NAMES))
    marker_csv = data_dir / "marker_metadata.csv"
    marker_df.to_csv(marker_csv, index=False)
    print(f"Saved marker metadata → {marker_csv}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORION preprocessing pipeline.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to ORION dataset directory."
    )
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir) / "ORIONCRC_dataset_tile_20x")
    main(Path(args.data_dir))
