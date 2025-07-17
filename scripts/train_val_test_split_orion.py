"""Script to split ORION-CRC patch-level dataframe into train, validation, and test sets."""

from pathlib import Path
import pandas as pd
import argparse

TEST_SLIDES = ['19510_C11_US_SCAN_OR_001__151039-registered.ome',
               '18459_LSP10364_US_SCAN_OR_001__092347-registered.ome']
VAL_SLIDES = ['19510_C19_US_SCAN_OR_001__153041-registered.ome',
              '19510_C30_US_SCAN_OR_001__155702-registered.ome']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ORION-CRC patch-level dataframe into train, validation, and test sets.")
    parser.add_argument("--dataframe_csv_path", type=str, required=True,
                        help="Path to the input dataframe CSV file.")
    parser.add_argument("--dataframe_folder_save", type=str, required=True,
                        help="Folder to save the split dataframes.")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.dataframe_csv_path)

    train_dataframe = dataframe[~dataframe["in_slide_name"].isin(VAL_SLIDES + TEST_SLIDES)]
    val_dataframe = dataframe[dataframe["in_slide_name"].isin(VAL_SLIDES)]
    test_dataframe = dataframe[dataframe["in_slide_name"].isin(TEST_SLIDES)]

    print(len(train_dataframe), len(val_dataframe), len(test_dataframe))
    save_folder = Path(args.dataframe_folder_save)
    save_folder.mkdir(exist_ok=True)
    train_dataframe.to_csv(str(save_folder / "train_dataframe.csv"), index=False)
    val_dataframe.to_csv(str(save_folder / "val_dataframe.csv"), index=False)
    test_dataframe.to_csv(str(save_folder / "test_dataframe.csv"), index=False)
