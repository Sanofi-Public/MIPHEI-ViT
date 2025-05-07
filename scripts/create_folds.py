import pandas as pd
from pathlib import Path
import numpy as np
import argparse

def extract_folds(dataframe, k):
    """
    Splits a 1D NumPy array into k roughly equal parts.
    
    Args:
        arr (np.ndarray): 1D array to be split.
        k (int): Number of parts to split the array into.
    
    Returns:
        list: A list of k NumPy arrays as the parts of the original array.
    """
    length = len(dataframe)
    arr = np.arange(length)
    np.random.shuffle(arr)
    # Calculate the base size for each part
    part_size = length // k
    remainder = length % k

    # Create a list of sizes for each part, distributing the remainder (if any)
    sizes = [part_size + (1 if i < remainder else 0) for i in range(k)]

    # Create the splits based on calculated sizes
    parts = []
    start = 0
    for size in sizes:
        parts.append(arr[start:start + size])
        start += size

    dataframe_folds = [dataframe.iloc[part].sort_index() for part in parts]
    return dataframe_folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataframes to create folds")
    parser.add_argument("--root_folder", type=str,
                        help="Path to the root folder containing train and val dataframes.")
    parser.add_argument("--sample_percent", type=float, default=0.3,
                        help="Percent of the data to keep for global dataset")
    parser.add_argument("--n_folds", type=int, default=3,
                        help="Number of folds")
    args = parser.parse_args()

    train_dataframe_path = str(Path(args.root_folder) / "train_dataframe.csv")
    train_dataframe = pd.read_csv(train_dataframe_path)
    val_dataframe_path = str(Path(args.root_folder) / "val_dataframe.csv")
    val_dataframe = pd.read_csv(val_dataframe_path)

    train_dataframe_sample = []
    for _, df_slide in train_dataframe.groupby("in_slide_name"):
        train_dataframe_sample.append(df_slide.sample(frac=args.sample_percent))
    train_dataframe_sample = pd.concat(train_dataframe_sample)

    train_folds = extract_folds(train_dataframe_sample, k=args.n_folds)

    val_dataframe_sample = []
    for _, df_slide in val_dataframe.groupby("in_slide_name"):
        val_dataframe_sample.append(df_slide.sample(frac=args.sample_percent))
    val_dataframe_sample = pd.concat(val_dataframe_sample)

    for idx_fold, train_fold in enumerate(train_folds):
        train_fold.to_csv(str(Path(args.root_folder) / f"train_dataframe_fold_{idx_fold}.csv"), index=False)
    val_dataframe_sample.to_csv(str(Path(args.root_folder) / "val_dataframe_fold.csv"), index=False)
