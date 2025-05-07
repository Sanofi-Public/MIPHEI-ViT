import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import json


import sys
sys.path.append('../')
from src.dataset import TileImg2ImgSlideDataset



def get_mean_stds(dataframe, channel_names):

    num_workers = os.cpu_count() - 1

    dataset_if = TileImg2ImgSlideDataset(dataframe,)
    dataloader_if = torch.utils.data.DataLoader(
            dataset_if, batch_size=32, shuffle=False,
            num_workers=num_workers, drop_last=False)

    num_channels = len(channel_names)
    sum_pixels_if = np.zeros(num_channels, dtype=np.float64)
    sum_squares_if = np.zeros(num_channels, dtype=np.float64)

    sum_pixels_he = np.zeros(3, dtype=np.float64)
    sum_squares_he = np.zeros(3, dtype=np.float64)

    pixel_pos_count = np.zeros(1, dtype=np.float64)

    for batch in tqdm(dataloader_if, total=len(dataloader_if)):
        batch_image = batch["image"].permute((0, 2, 3, 1)).numpy()
        batch_target = batch["target"].permute((0, 2, 3, 1)).numpy()


        batch_target = batch_target / 255
        sum_pixels_if += np.sum(batch_target, axis=(0, 1, 2))
        sum_squares_if += np.sum(batch_target**2, axis=(0, 1, 2))

        batch_image = batch_image / 255
        sum_pixels_he += np.sum(batch_image, axis=(0, 1, 2))
        sum_squares_he += np.sum(batch_image**2, axis=(0, 1, 2))
        pixel_pos_count += batch_image.shape[0] * batch_image.shape[1] * batch_image.shape[2]

    mean_if = sum_pixels_if / pixel_pos_count
    variance_if = (sum_squares_if / pixel_pos_count) - (mean_if ** 2)
    std_if = np.sqrt(variance_if)
    mean_if, std_if = mean_if * 255, std_if * 255

    mean_he = sum_pixels_he / pixel_pos_count
    variance_he = (sum_squares_he / pixel_pos_count) - (mean_he ** 2)
    std_he = np.sqrt(variance_he)
    mean_he, std_he = mean_he * 255, std_he * 255

    channel_stats = {}
    for idx_channel, channel_name in enumerate(channel_names):
        channel_stats[channel_name] = {
            "mean": mean_if[idx_channel].tolist(), "std": std_if[idx_channel].tolist(),
            "channel_idx": idx_channel}
    
    channel_stats["RGB"] = {"mean": mean_he.tolist(), "std": std_he.tolist()}
    return channel_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get mean and std for channels")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to the tile dataframe")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--channel_names", type=str, nargs='+', required=True, help="List of channel names")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.dataframe_path)
    output_path = args.output_path
    channel_names = args.channel_names

    channel_stats = get_mean_stds(dataframe, channel_names)
    with open(output_path, "w") as f:
        json.dump(channel_stats, f, indent=4)
