import pyvips
from slidevips import SlideVips
from slidevips.torch_datasets import Img2ImgSlideDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops_table
from pathlib import Path
import os
import gc


def dataloader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


def get_tile_positions(slide_dims, tile_size):
    tile_positions = np.meshgrid(
        np.arange(0, slide_dims[0], tile_size),
        np.arange(0, slide_dims[1], tile_size))
    tile_positions = np.stack(tile_positions, axis=-1).reshape((-1, 2))
    return tile_positions


def extract_expression_matrix(targ_slide_path, nuclei_slide_path, output_dir, level=0, tile_size=8192):

    pyvips.cache_set_max(0)
    in_slide_name = Path(targ_slide_path).stem
    slide_dataframe = pd.DataFrame(columns=["in_slide_name", "in_slide_path", "targ_slide_path"])
    slide_dataframe["in_slide_name"] = [in_slide_name]
    slide_dataframe["in_slide_path"] = [targ_slide_path]
    slide_dataframe["targ_slide_path"] = [nuclei_slide_path]

    slide = SlideVips(targ_slide_path, mode="IF")
    slide_dims = slide.dimensions
    n_channels = slide.n_channels
    tile_positions = get_tile_positions(slide_dims, tile_size)
    dataframe = pd.DataFrame(columns=["in_slide_name", "x", "y", "level", "tile_size_x", "tile_size_y"])
    dataframe["in_slide_name"] = [in_slide_name] * len(tile_positions)
    dataframe["x"] = tile_positions[..., 0]
    dataframe["y"] = tile_positions[..., 1]
    dataframe["level"] = level
    dataframe["tile_size_x"] = tile_size
    dataframe["tile_size_y"] = tile_size

    num_workers = int(os.cpu_count() * 2/3)
    aggregation_rules = {'area': 'sum', 'intensity_sum': 'sum',
                         'X_centroid': 'mean', 'Y_centroid': 'mean'}

    df_wsi = None
    for idx_channel in tqdm(range(n_channels),
                            desc="Channel Single Cell Extraction"): # avoid RAM problems
        dataset = Img2ImgSlideDataset(
            slide_dataframe, dataframe, mode_in="IF", mode_targ="IF",
            in_channel_idxs=[idx_channel], targ_channel_idxs=[0])
        dataloader = DataLoader(dataset, shuffle=False, drop_last=False,
                                batch_size=1, num_workers=num_workers,
                                worker_init_fn=dataloader_worker_init_fn)

        df_channel = []
        if df_wsi is None:
            properties = ['label', 'area', 'intensity_image', 'centroid']
        else:
            properties = ['label', 'area', 'intensity_image']
        for batch in dataloader:
            image_if = np.uint16(batch["image"][0, 0].numpy())
            image_nuclei = np.int32(batch["target"][0, 0].numpy())
            props = regionprops_table(
                image_nuclei,
                intensity_image=image_if,
                properties=properties
            )
            df_roi = pd.DataFrame(props)
            df_roi['intensity_sum'] = df_roi['intensity_image'].apply(np.sum)
            df_roi = df_roi.drop(columns=['intensity_image'])
            if df_wsi is None:
                tile_name = batch["tile_name"][0]
                x, y = tuple(map(int, tile_name.split("_")[-5:-3]))
                df_roi.rename(columns={'centroid-0': 'Y_centroid', 'centroid-1': 'X_centroid'}, inplace=True)
                df_roi["X_centroid"] += x
                df_roi["Y_centroid"] += y
            df_roi["area"] = df_roi["area"].astype(np.int32)
            df_channel.append(df_roi)

        df_channel = pd.concat(df_channel, ignore_index=True)
        df_channel = df_channel.groupby('label', as_index=False).agg(aggregation_rules)
        df_channel["intensity_mean"] = df_channel["intensity_sum"] / df_channel["area"]
        df_channel = df_channel.drop(columns=["intensity_sum"])
        df_channel = df_channel.rename(columns={"intensity_mean": idx_channel})
        if df_wsi is None:
            df_wsi = df_channel
            del aggregation_rules["X_centroid"], aggregation_rules["Y_centroid"]
        else:
            assert len(df_wsi) == len(df_channel)
            df_wsi = pd.merge(df_wsi, df_channel, on='label', suffixes=("", "_y"))
            assert df_wsi.isna().sum().sum() == 0
            df_wsi = df_wsi.drop(columns=["area_y"])
        del df_channel, dataloader, dataset, image_if, image_nuclei
        gc.collect()

    Path(output_dir).mkdir(exist_ok=True)
    out_csv_path = str(Path(output_dir) / f"{in_slide_name}.csv")
    df_wsi.to_csv(out_csv_path, index=False)
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract expression matrix from WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the csv files will be saved")

    parser.add_argument("--level", type=int, default=0, help="Level used during single cell expression extraction")
    parser.add_argument("--tile_size", type=int, default=8192, help="Tile size used during single cell expression extraction")
    args = parser.parse_args()

    slide_dataframe_path = args.slide_dataframe_path
    slide_dataframe = pd.read_csv(slide_dataframe_path)
    output_dir = args.output_dir
    level = args.level
    tile_size = args.tile_size

    for _, row in slide_dataframe.iterrows():
        targ_slide_path = row["targ_slide_path"]
        nuclei_slide_path = row["nuclei_slide_path"]
        out_csv_path = extract_expression_matrix(targ_slide_path, nuclei_slide_path, output_dir, level=level, tile_size=tile_size)
        gc.collect()
