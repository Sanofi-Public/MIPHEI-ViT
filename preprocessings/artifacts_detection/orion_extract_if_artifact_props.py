import os
os.environ['VIPS_CONCURRENCY'] = '1'
import pandas as pd
import torch
import numpy as np
from slidevips.torch_datasets import SlideDataset
from pathlib import Path
import ome_types

from tqdm import tqdm
import gc


def dataloader_worker_init_fn(worker_id):
    import pyvips
    pyvips.cache_set_max(0)
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function

# global variables specific to ORION dataset
# You need to adapt this script if you want to adapt to your own dataset
ARTIFACT_THRESHOLD = 2000
ARTIFACT_NAME = "Blank"
ARTIFACT_PERCENT_THRESHOLD = 0.05


def extract_artifact_props(slide_dataframe, dataframe, output_props_path):

    slide_dataframe = slide_dataframe.copy()
    slide_dataframe["in_slide_path"] = slide_dataframe["targ_slide_path"] # if input not H&E

    artifact_percent_info = []
    for slide_name, dataframe_slide in tqdm(
            dataframe.groupby("in_slide_name", sort=False), 
            total=dataframe["in_slide_name"].nunique(), 
            desc="Slides", leave=True):

        slide_path = slide_dataframe.loc[slide_dataframe["in_slide_name"] == slide_name, "in_slide_path"].values[0]
        ome_metadata = ome_types.from_tiff(slide_path)
        channel_names = [channel.name for channel in ome_metadata.images[0].pixels.channels]

        artifact_channel_idx = channel_names.index(ARTIFACT_NAME)

        original_indices = dataframe_slide.index.tolist()
        dataset = SlideDataset(slide_dataframe, dataframe_slide,
                               mode="IF", channel_idxs=[artifact_channel_idx])

        num_workers = os.cpu_count() - 1
        batch_size = 64
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers
        )

        artifact_percent_slide = []
        for batch in tqdm(dataloader, total=len(dataloader), leave=False):
            batch_artifact = batch["image"]
            batch_artifact_percentages = torch.mean((batch_artifact > ARTIFACT_THRESHOLD).float(), dim=(1, 2, 3))
            artifact_percent_slide.append(batch_artifact_percentages.numpy())
        artifact_percent_slide = np.hstack(artifact_percent_slide)
        artifact_percent_info.append((original_indices, artifact_percent_slide))
        gc.collect()

    # Reconstruct global artifact_percent array
    artifact_percent = np.zeros(len(dataframe), dtype=np.float32)

    for indices, artifact_percent_slide in artifact_percent_info:
        artifact_percent[indices] = artifact_percent_slide

    np.save(output_props_path, artifact_percent)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide dataframe")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to the input tile dataframe")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output artifact proportion npy file")
    args = parser.parse_args()

    slide_dataframe = pd.read_csv(args.slide_dataframe_path)
    dataframe = pd.read_csv(args.dataframe_path)
    output_path = args.output_path

    extract_artifact_props(slide_dataframe, dataframe, output_path)
