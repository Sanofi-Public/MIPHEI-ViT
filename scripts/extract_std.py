import pandas as pd
import torch
from tqdm import tqdm
import json

import sys
sys.path.append("../")
from src.dataset import DataModule


if __name__ == "__main__":
    slide_dataframe = pd.read_csv("/root/workdir/data/slide_dataframe.csv")
    train_dataframe = pd.read_csv("/root/workdir/data/train_dataframe.csv")
    val_dataframe = pd.read_csv("/root/workdir/data/val_dataframe.csv")
    test_dataframe = pd.read_csv("/root/workdir/data/test_dataframe.csv")
    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe, 
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=None, from_slide=False,
        batch_size=32, pin_memory=False,
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
            sum_squares_channels = torch.zeros(n_channels, device=target.device, dtype=torch.float64)

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
    with open("../stats.json", "w") as f:
        json.dump(data_json, f)
