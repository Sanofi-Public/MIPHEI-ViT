import pandas as pd
from pathlib import Path
import os


if __name__ == "__main__":

    train_dataframe = pd.read_csv("/root/workdir/data/train_dataframe.csv")
    val_dataframe = pd.read_csv("/root/workdir/data/val_dataframe.csv")

    parent_folder = Path(train_dataframe["image_path"].iloc[0]).parent.parent


    trainA_dir = parent_folder / "trainA"
    trainB_dir = parent_folder / "trainB"


    for _, row in train_dataframe.iterrows():
        image_path = row["image_path"]
        target_path = row["target_path"]
        new_target_path = str(trainB_dir / Path(target_path).name)
        new_new_target_path = str(trainB_dir / (Path(image_path).stem + '.tiff'))
        os.rename(new_target_path, new_new_target_path)

    valA_dir = parent_folder / "valA"
    valB_dir = parent_folder / "valB"

    for _, row in val_dataframe.iterrows():
        image_path = row["image_path"]
        target_path = row["target_path"]
        new_target_path = str(valB_dir / Path(target_path).name)
        new_new_target_path = str(valB_dir / (Path(image_path).stem + '.tiff'))
        os.rename(new_target_path, new_new_target_path)
