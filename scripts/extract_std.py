"""
Script to compute per-channel mean and standard deviation of train set targets for Weighted MSE.

The computed statistics are saved as a JSON file for use in model training.
"""

import argparse
import json
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("../")
from src.dataset import DataModule


def extract_std(slide_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame,
                val_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, output_path: str,
                batch_size: int = 32) -> None:
    """
    Compute per-channel standard deviation of training set targets for Weighted MSE Loss.

    This function iterates over the training data, accumulating the sum and sum of squares for each
    channel in the target mIF images. It then computes the mean and standard deviation for each
    channel, scaling the results back to the original [0, 255] range. The computed statistics are
    saved to the specified output path in JSON format. The extracted standard deviations are
    typically used for weighting the MSE loss during model training.
    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing slide-level metadata.
        train_dataframe (pd.DataFrame): DataFrame containing training data information.
        val_dataframe (pd.DataFrame): DataFrame containing validation data information.
        test_dataframe (pd.DataFrame): DataFrame containing test data information.
        output_path (str): Path to the output JSON file where the mean and standard deviation will
            be saved.
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
    Returns:
        None: The function saves the computed statistics to a file and does not return anything.
    """
    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe,
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=None, from_slide=False,
        batch_size=batch_size, pin_memory=False,
        return_nuclei=False, train_sampler=None,
        preprocess_input_fn=None, preprocess_target_fn=None,
        spatial_augmentations=None, color_augmentations=None)
    data_module.setup()
    train_dataloader, _, _ = data_module.get_dataloaders()

    sum_channels = None
    sum_squares_channels = None
    n_pixels = 0

    # Iterate over batches
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        target = batch["target"] / 255

        batch_size, n_channels, width, height = target.shape
        num_elements = batch_size * width * height  # Total pixels in this batch

        if sum_channels is None:  # Initialize on the first batch
            sum_channels = torch.zeros(n_channels, device=target.device, dtype=torch.float64)
            sum_squares_channels = torch.zeros(n_channels, device=target.device,
                                               dtype=torch.float64)

        # Sum and sum of squares across the batch
        sum_channels += target.sum(dim=[0, 2, 3])  # Sum over batch, width, height
        sum_squares_channels += (target ** 2).sum(dim=[0, 2, 3])  # Sum of squares

        # Keep track of total pixel count
        n_pixels += num_elements

    # Compute mean and std for each channel
    mean_channels = sum_channels / n_pixels
    std_channels = torch.sqrt((sum_squares_channels / n_pixels) - mean_channels ** 2)
    mean_channels = mean_channels * 255
    std_channels = std_channels * 255

    mean_channels = mean_channels.numpy().tolist()
    std_channels = std_channels.numpy().tolist()
    data_json = {"mean": mean_channels, "std_channels": std_channels}
    with open(output_path, "w") as f:
        json.dump(data_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-channel std from train set targets.")
    parser.add_argument("--slide_dataframe_path", type=str, required=True,
                        help="Path to slide_dataframe.csv")
    parser.add_argument("--train_dataframe_path", type=str, required=True,
                        help="Path to train_dataframe.csv")
    parser.add_argument("--val_dataframe_path", type=str, required=True,
                        help="Path to val_dataframe.csv")
    parser.add_argument("--test_dataframe_path", type=str, required=True,
                        help="Path to test_dataframe.csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")

    args = parser.parse_args()

    slide_dataframe = pd.read_csv(args.slide_dataframe_path)
    train_dataframe = pd.read_csv(args.train_dataframe_path)
    val_dataframe = pd.read_csv(args.val_dataframe_path)
    test_dataframe = pd.read_csv(args.test_dataframe_path)

    extract_std(
        slide_dataframe,
        train_dataframe,
        val_dataframe,
        test_dataframe,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
    extract_std(slide_dataframe, train_dataframe, val_dataframe, test_dataframe)
