import os
import pyvips

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import os
from tqdm import tqdm
import cv2
import json
from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu
from slidevips.torch_datasets import SlideDataset
from slidevips.ome_metadata import adapt_ome_metadata
import tempfile
import gc


def find_percentile_bin(histogram, percentile):
    """
    Find the bin index corresponding to the given percentile in a normalized histogram.
    
    Args:
    histogram (numpy.ndarray): The normalized histogram (sum equals 1).
    percentile (float): The desired percentile (0-100).
    
    Returns:
    int: The bin index corresponding to the percentile.
    """
    # Ensure the percentile is between 0 and 100
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    
    percentile_values = []
    for histogram_channel in histogram:
        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(histogram_channel)
        # Find the first index where the CDF value is greater than or equal to the percentile
        bin_index_channel = np.searchsorted(cdf, percentile, side='right')
        percentile_values.append(bin_index_channel)
    return np.asarray(percentile_values)


def get_full_tile_positions(slide_dims, tile_size):
    tile_positions = np.meshgrid(
        np.arange(0, slide_dims[0], tile_size),
        np.arange(0, slide_dims[1], tile_size))
    tile_positions = np.stack(tile_positions, axis=-1).reshape((-1, 2))
    return tile_positions


def dataloader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


def apply_cleaning_wsi(in_slide_path, global_hists, lambdas_path, channel_names, af_channel_name,
                       output_dir, drop_channel_names=[], batch_size=16, tile_size=2048):

    in_slide_name = Path(in_slide_path).stem

    pyvips.cache_set_max(0)
    with open(lambdas_path, "r") as f:
        settings = json.load(f)
    lambda_factors = []
    biases = []
    max_values = []
    if af_channel_name not in drop_channel_names:
        drop_channel_names.append(af_channel_name)
    channel_idxs = [i for i, channel_name in enumerate(channel_names) if channel_name not in drop_channel_names]
    final_channel_names = [channel_name for channel_name in channel_names if channel_name not in drop_channel_names]
    for i, channel_idx in enumerate(channel_idxs):
        settings_channel = settings[str(channel_idx)]
        lambda_factors.append(settings_channel["lambda"])
        biases.append(settings_channel["bias"])
        curr_hist = global_hists[i].copy()
        norm_hist = np.expand_dims(curr_hist / curr_hist.sum(), axis=0).copy()
        max_value = find_percentile_bin(norm_hist, percentile=0.99).item()
        max_values.append(max_value)
    
    lambda_factors = np.asarray(lambda_factors)
    biases = np.asarray(biases)
    max_values = np.asarray(max_values)

    slide_dataframe = pd.DataFrame(
        columns=["in_slide_name", "in_slide_path"],
        data=[[in_slide_name, in_slide_path]])

    af_idx = channel_names.index(af_channel_name)

    slide_if = SlideVips(in_slide_path, mode="IF", channel_idxs=channel_idxs +[af_idx])
    resolution = slide_if.mpp
    magnification = slide_if.magnification
    slide_dims = slide_if.dimensions
    tile_positions = get_full_tile_positions(slide_dims, tile_size)
    dataframe = pd.DataFrame()
    dataframe["in_slide_name"] = [in_slide_name] * len(tile_positions)
    dataframe["x"] = tile_positions[..., 0]
    dataframe["y"] = tile_positions[..., 1]
    dataframe["level"] = 0
    dataframe["tile_size_x"] = tile_size
    dataframe["tile_size_y"] = tile_size
    slide_if.close()
    num_workers = 0# 2


    channel_temp_paths = []
    n_channels = len(channel_idxs)
    for i in tqdm(range(n_channels), desc="WSI Cleaning", leave=False):
        idx_channel = channel_idxs[i]
        lambda_factor = lambda_factors[i]
        bias = biases[i]
        max_value = max_values[i]

        dataset = SlideDataset(
            slide_dataframe,
            dataframe,
            mode="IF",
            channel_idxs=[idx_channel, af_idx],
            scale_factor=1.
        )
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, drop_last=False,
                worker_init_fn=dataloader_worker_init_fn, pin_memory=False,)

        final_img = pyvips.Image.black(slide_dims[0], slide_dims[1], bands=1).cast(pyvips.BandFormat.UCHAR)

        for idx_batch, batch in enumerate(dataloader):
            tile_name_batch = batch["tile_name"]
            tile_positions_batch = [tuple(map(int, tile_name.split("_")[-5:-3])) for tile_name in tile_name_batch]
            batch = batch["image"].float().numpy()
            out_batch = np.maximum(
                np.float32(batch[:, 0] - lambda_factor * batch[:, 1] + bias),
                0.)
            out_batch = np.uint8(np.clip(np.log1p(out_batch / max_value), 0., 1.) * 255)
           

            for tile_position, out in zip(tile_positions_batch, out_batch):
                tile = pyvips.Image.new_from_array(out)
                final_img = final_img.insert(tile,
                    tile_position[0], 
                    tile_position[1])

        del tile, out_batch
        dataset.reset()
        del dataloader, dataset
        temp_path = str(Path(tempfile.gettempdir()) / f"{i}_{in_slide_name}.ome.tiff")
        final_img.write_to_file(temp_path)
        channel_temp_paths.append(temp_path)
        del final_img
        gc.collect()


    output_path = str(Path(output_dir) / (in_slide_name.replace(".ome", "") + ".ome.tiff"))
    channel_imgs = []
    for channel_path in channel_temp_paths:
        channel_img = pyvips.Image.new_from_file(channel_path, access="sequential").colourspace("b-w")
        channel_imgs.append(channel_img)
    final_img = pyvips.Image.arrayjoin(channel_imgs, across=1).colourspace("b-w")
    final_img = final_img.copy()

    ome_xml_metadata = adapt_ome_metadata(final_img, resolution, final_channel_names, magnification)
    image_height = final_img.height // n_channels
    final_img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    final_img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)
    Path(output_dir).mkdir(exist_ok=True)
    final_img.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / resolution,
        yres=1000 / resolution,
        page_height=image_height)

    for channel_path in channel_temp_paths:
        os.remove(channel_path)
    return output_path


def extract_histogram(slide_dataframe, level, tile_size, lambdas_path, channel_names, af_channel_name,
                      artifact_channel_name=None, artifact_treshold=2000):

    slide_dataframe = slide_dataframe.copy()
    slide_dataframe["in_slide_path"] = slide_dataframe["targ_slide_path"]  # Trick: input is mIF not H&E

    dataframe = []
    dtype = None
    for _, row in tqdm(slide_dataframe.iterrows(), total=len(slide_dataframe), 
                       desc="Histogram Tiling", leave=False):
        slide = SlideVips(row["in_slide_path"], mode="IF")
        thumbnail = slide.get_thumbnail((3000, 3000))
        thumbnail_dtype = thumbnail.dtype
        thumbnail = thumbnail.mean(axis=-1, keepdims=True)
        if dtype is None:
            dtype = thumbnail_dtype
            thumbnail = np.uint16(thumbnail)
        else:
            assert dtype == thumbnail_dtype
            thumbnail = np.uint8(thumbnail)

        try:
            tile_postions, _ = get_locs_otsu(thumbnail, slide.dimensions, tile_size)
        except ZeroDivisionError:
            slide.close()
            continue
        slide.close()
        dataframe_slide = pd.DataFrame()
        dataframe_slide["in_slide_name"] = [row["in_slide_name"]] * len(tile_postions)
        dataframe_slide["x"] = tile_postions[..., 0]
        dataframe_slide["y"] = tile_postions[..., 1]
        dataframe_slide["level"] = level
        dataframe_slide["tile_size_x"] = tile_size
        dataframe_slide["tile_size_y"] = tile_size
        dataframe.append(dataframe_slide)

    dataframe = pd.concat(dataframe)
    dataframe = (
        dataframe.groupby('in_slide_name', group_keys=False)
        .apply(lambda x: x.sample(n=500, random_state=42) if len(x) >= 500 else x)
    )
    print(len(dataframe))
    is_uint16 = dtype == np.uint16

    with open(lambdas_path, "r") as f:
        settings = json.load(f)
    af_idx = channel_names.index(af_channel_name)

    num_workers = 0
    global_hists = []
    channel_idxs = [i for i, channel_name in enumerate(channel_names) if channel_name != af_channel_name]
    for idx, idx_channel in tqdm(enumerate(channel_idxs),total=len(channel_idxs),
                                 desc="Histogram Computation"):
        settings_channel = settings[str(idx_channel)]
        lambda_factor = settings_channel["lambda"]
        bias = settings_channel["bias"]


        if artifact_channel_name is not None:
            artifact_idx = channel_names.index(artifact_channel_name)
            additional_idxs = [artifact_idx, af_idx]
        else:
            additional_idxs = [af_idx]
        dataset_if = SlideDataset(slide_dataframe, dataframe, channel_idxs=[idx_channel] + additional_idxs, mode="IF", scale_factor=1.)
        dataloader_if = torch.utils.data.DataLoader(
                dataset_if, batch_size=32, shuffle=False,
                num_workers=num_workers, drop_last=False)

        if is_uint16:
            global_hist = np.zeros(65536, dtype=np.float64)
        else:
            global_hist = np.zeros(256, dtype=np.float64)

        for batch in tqdm(dataloader_if, total=len(dataloader_if),
                          desc="Channel histogram computation", leave=False):
            batch = batch["image"]
            batch = batch.float().numpy()
            if artifact_channel_name is not None:
                batch_artifact = batch[:, 1]
            batch = batch[:, 0] - lambda_factor * batch[:, -1] + bias
            if is_uint16:
                batch = np.uint16(np.maximum(batch, 0))
            else:
                batch = np.uint8(np.maximum(batch, 0))
            #batch = np.where(batch > background_treshes, batch, 0)
            width_tile = batch.shape[2]
            batch = batch.reshape((-1, width_tile))
            mask = (batch > 0)
            if artifact_channel_name is not None:
                mask = mask * (batch_artifact.reshape((-1, width_tile)) < artifact_treshold)
            if np.sum(mask) > 0:
                if is_uint16:
                    hist = np.float64(cv2.calcHist([batch], [0], np.uint8(mask) * 255,
                                                   [65536], [0, 65536]).flatten())
                else:
                    hist = np.float64(cv2.calcHist([batch], [0], np.uint8(mask) * 255,
                                                   [256], [0, 256]).flatten())
                global_hist += hist

        #global_hist = global_hist / global_hist.sum()
        global_hists.append(global_hist)
        #max_values.append(find_percentile_bin(global_hist, percentile=0.999).reshape((1, -1, 1, 1)))

    global_hists = np.vstack(global_hists)
    #np.save("hist.npy", global_hists)
    return global_hists

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the cleaned slides will be saved")
    parser.add_argument("--channel_names", type=str, nargs="+", required=True, help="List of channel names")
    parser.add_argument("--af_channel_name", type=str, required=True, help="Name of the AF channel")
    parser.add_argument("--lambdas_path", type=str, required=True, help="Path to the lambda settings JSON file")
    parser.add_argument("--artifact_channel_name", type=str, required=True, help="Name of the artifact channel")
    parser.add_argument("--artifact_treshold", type=int, default=2000, help="Threshold for the artifact channel")
    parser.add_argument("--drop_channel_names", type=str, nargs="+", default=[], help="List of channel names to drop")

    args = parser.parse_args()

    slide_dataframe_path = args.slide_dataframe_path
    slide_dataframe = pd.read_csv(slide_dataframe_path)
    output_dir = args.output_dir
    channel_names = args.channel_names
    af_channel_name = args.af_channel_name
    lambdas_path = args.lambdas_path
    artifact_channel_name = args.artifact_channel_name
    artifact_treshold = args.artifact_treshold
    drop_channel_names = args.drop_channel_names

    print("Step 1: Histogram extraction")
    global_hists = extract_histogram(slide_dataframe, level=0, tile_size=512,
                      lambdas_path=lambdas_path,
                      channel_names=channel_names,
                      af_channel_name=af_channel_name,
                      artifact_channel_name=artifact_channel_name, artifact_treshold=artifact_treshold)
    #np.save("orion_hist.npy", global_hists)

    print("Step 2: Cleaning WSI")
    output_paths = []
    for _, row in tqdm(slide_dataframe.iterrows(), total=len(slide_dataframe),
                       desc="Cleaning WSI"):
        mif_slide_path = row["targ_slide_path"]
        output_path = apply_cleaning_wsi(
            mif_slide_path, global_hists=global_hists, lambdas_path=lambdas_path,
            channel_names=channel_names, af_channel_name=af_channel_name, output_dir=output_dir,
            drop_channel_names=drop_channel_names, batch_size=16, tile_size=2048)
        output_paths.append(output_path)
        gc.collect()

    slide_dataframe["targ_slide_path"] = output_paths
    clean_slide_dataframe_path = Path(slide_dataframe_path).parent / ("clean_" + Path(slide_dataframe_path).name)
    slide_dataframe["targ_slide_path"] = slide_dataframe["targ_slide_path"].apply(lambda x: str(Path(x).resolve()))
    slide_dataframe.to_csv(clean_slide_dataframe_path.resolve(), index=False)
