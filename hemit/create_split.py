import pandas as pd
from pathlib import Path
import os


if __name__ == "__main__":

    train_dataframe = pd.read_csv("/root/workdir/data/train_dataframe.csv")
    val_dataframe = pd.read_csv("/root/workdir/data/val_dataframe.csv")

    parent_folder = Path(train_dataframe["image_path"].iloc[0]).parent.parent


    trainA_dir = parent_folder / "trainA"
    trainB_dir = parent_folder / "trainB"
    trainA_dir.mkdir(exist_ok=False)
    trainB_dir.mkdir(exist_ok=False)

    for _, row in train_dataframe.iterrows():
        image_path = row["image_path"]
        new_image_path = str(trainA_dir / Path(image_path).name)
        target_path = row["target_path"]
        new_target_path = str(trainB_dir / (Path(image_path).stem + '.tiff'))
        os.rename(image_path, new_image_path)
        os.rename(target_path, new_target_path)

    valA_dir = parent_folder / "valA"
    valB_dir = parent_folder / "valB"
    valA_dir.mkdir(exist_ok=False)
    valB_dir.mkdir(exist_ok=False)

    for _, row in val_dataframe.iterrows():
        image_path = row["image_path"]
        new_image_path = str(valA_dir / Path(image_path).name)
        target_path = row["target_path"]
        new_target_path = str(valB_dir / (Path(image_path).stem + '.tiff'))
        os.rename(image_path, new_image_path)
        os.rename(target_path, new_target_path)
