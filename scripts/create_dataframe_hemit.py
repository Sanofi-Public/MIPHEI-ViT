"""Module to generate and save CSV dataframes with  metadata for the HEMIT dataset."""

import argparse
from pathlib import Path

import pandas as pd


def main(data_dir: str, dataframe_folder_save: Path) -> None:
    """
    Generate and save dataframes containing image paths and related metadata for HEMIT dataset for \
    training, validation, and test datasets.

    This function scans the specified directory for image files in the 'train/input', 'val/input',
    and 'test/input' subdirectories. For each image, it constructs corresponding paths for label,
    mask, and CSV files, and compiles this information into pandas DataFrames. The resulting
    DataFrames are saved as CSV files in the specified output folder.
    Args:
        data_dir (str or Path): Path to the root directory containing 'train', 'val', and 'test'
            subdirectories with 'input' images of HEMIT dataset.
        dataframe_folder_save (str or Path): Path to the directory where the generated CSV files
            will be saved.
    Raises:
        AssertionError: If any of the constructed image, label, or mask paths do not exist on disk.
    Outputs:
        Saves the following CSV files in `dataframe_folder_save`:
            - slide_dataframe_hemit.csv: Contains slide names and corresponding nuclei CSV paths.
            - dataframe_hemit.csv: Contains combined train, validation, and test data with image,
                label, and mask paths.
            - train_dataframe_hemit.csv: Contains training data only.
            - val_dataframe_hemit.csv: Contains validation data only.
            - test_dataframe_hemit.csv: Contains test data only.
    """
    dataframe_columns = ['in_slide_name', 'image_path', 'target_path', 'nuclei_path',
                         'nuclei_csv_path']
    train_dataframe = pd.DataFrame(columns=dataframe_columns)
    val_dataframe = pd.DataFrame(columns=dataframe_columns)
    test_dataframe = pd.DataFrame(columns=dataframe_columns)

    train_dataframe["image_path"] = [str(fn)
                                     for fn in (Path(data_dir) / "train/input").glob("*.tif")]
    train_dataframe["in_slide_name"] = train_dataframe["image_path"].apply(lambda x: Path(x).name)
    train_dataframe["target_path"] = train_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/label/"))
    train_dataframe["nuclei_path"] = train_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/"))

    val_dataframe["image_path"] = [str(fn) for fn in (Path(data_dir) / "val/input").glob("*.tif")]
    val_dataframe["in_slide_name"] = val_dataframe["image_path"].apply(lambda x: Path(x).name)
    val_dataframe["target_path"] = val_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/label/"))
    val_dataframe["nuclei_path"] = val_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/"))

    test_dataframe["image_path"] = [str(fn) for fn in (Path(data_dir) / "test/input").glob("*.tif")]
    test_dataframe["in_slide_name"] = test_dataframe["image_path"].apply(lambda x: Path(x).name)
    test_dataframe["target_path"] = test_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/label/"))
    test_dataframe["nuclei_path"] = test_dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/mask/"))

    print(len(train_dataframe), len(val_dataframe), len(test_dataframe))

    dataframe = pd.concat((train_dataframe, val_dataframe, test_dataframe))
    slide_dataframe = pd.DataFrame()
    slide_dataframe["in_slide_name"] = dataframe["in_slide_name"]
    slide_dataframe["nuclei_csv_path"] = dataframe["image_path"].apply(
        lambda x: x.replace("/input/", "/csv/").replace(".tif", ".csv"))

    assert dataframe["image_path"].apply(lambda x: Path(x).exists()).all()
    assert dataframe["target_path"].apply(lambda x: Path(x).exists()).all()
    assert dataframe["nuclei_path"].apply(lambda x: Path(x).exists()).all()

    slide_dataframe.to_csv(str(dataframe_folder_save / "slide_dataframe_hemit.csv"), index=False)
    dataframe.to_csv(str(dataframe_folder_save / "dataframe_hemit.csv"), index=False)
    train_dataframe.to_csv(str(dataframe_folder_save / "train_dataframe_hemit.csv"), index=False)
    val_dataframe.to_csv(str(dataframe_folder_save / "val_dataframe_hemit.csv"), index=False)
    test_dataframe.to_csv(str(dataframe_folder_save / "test_dataframe_hemit.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataframes for HEMIT dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--dataframe_folder_save", type=str, required=True,
                        help="Folder to save the dataframes.")
    args = parser.parse_args()

    main(args.data_dir, Path(args.dataframe_folder_save))
