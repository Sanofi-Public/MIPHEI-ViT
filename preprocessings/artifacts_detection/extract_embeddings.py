import os
os.environ['VIPS_CONCURRENCY'] = '1'
import pandas as pd
import torch
import numpy as np
from slidevips.torch_datasets import SlideDataset
import albumentations as A
import shutil
import tempfile

from tqdm import tqdm
import gc


import sys
sys.path.append('../../')
from src.generators.foundation_models import FOUNDATION_MODEL_REGISTRY
from src.dataset import NormalizationLayer


def dataloader_worker_init_fn(worker_id):
    import pyvips
    pyvips.cache_set_max(0)
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


def extract_embeddings(slide_dataframe, dataframe, output_emb_path, model_name="hoptimus0", downsample_2x=True):
    tile_size = dataframe["tile_size_x"].iloc[0]
    if downsample_2x:
        inference_tile_size = tile_size//2
        transforms = A.Resize(inference_tile_size, inference_tile_size, interpolation=1)
    else:
        inference_tile_size = tile_size
        transforms = A.NoOp()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    global_pool = "avg" if model_name == "ctranspath" else "token"
    model = FOUNDATION_MODEL_REGISTRY[model_name](img_size=inference_tile_size, pretrained=True,
                                                  global_pool=global_pool).eval().to(device).half()

    if model_name == "hoptimus0":
        mean, std = np.asarray([0.707223, 0.578729, 0.703617])*255, np.asarray([0.211883, 0.230117, 0.177517])*255
    else:
        mean, std = np.asarray([0.485, 0.456, 0.406])*255, np.asarray([0.229, 0.224, 0.225])*255

    channel_stats_rgb = {
        "mean": mean,  # *255 for NormalizationLayer
        "std": std, # *255 for NormalizationLayer
    }
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb)

    temp_dir = tempfile.mkdtemp(prefix="temp_embeddings_")
    embeddings_info = []

    for slide_name, dataframe_slide in tqdm(
            dataframe.groupby("in_slide_name", sort=False), 
            total=dataframe["in_slide_name"].nunique(), 
            desc="Slides Inference", leave=True):
        original_indices = dataframe_slide.index.tolist()
        dataset = SlideDataset(slide_dataframe, dataframe_slide,
                            preprocess_input_fn=preprocess_input_fn,
                            spatial_augmentations=transforms)

        num_workers = os.cpu_count() - 1
        batch_size = 64
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers
        )

        embeddings_slide = []
        for batch in tqdm(dataloader, total=len(dataloader), leave=False):
            with torch.inference_mode():
                embeddings_batch = model(batch["image"].to(device).half()).cpu()
            embeddings_slide.append(embeddings_batch.numpy())
        embeddings_slide = np.concatenate(embeddings_slide, axis=0)
        slide_tmp_path = os.path.join(temp_dir, f"{slide_name}_embeddings.npy")
        np.save(slide_tmp_path, embeddings_slide)
        embeddings_info.append((original_indices, slide_tmp_path))
        gc.collect()

    # Reconstruct global embedding array
    embeddings = np.zeros((len(dataframe), embeddings_slide.shape[1]), dtype=np.float16)

    for indices, path in embeddings_info:
        emb = np.load(path)
        embeddings[indices] = emb

    np.save(output_emb_path, embeddings)
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide dataframe")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to the input tile dataframe")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output embeddings npy file")
    parser.add_argument("--model_name", type=str, default="hoptimus0",
                        help="Name of the foundation model use to extract embeddings")
    parser.add_argument("--downsample_2x", action="store_true",
                        help="Enable 2x downsampling, for faster inference or to match 20x magnification")
    args = parser.parse_args()

    slide_dataframe = pd.read_csv(args.slide_dataframe_path)
    dataframe = pd.read_csv(args.dataframe_path)
    output_path = args.output_path
    model_name = args.model_name
    downsample_2x = args.downsample_2x

    extract_embeddings(slide_dataframe, dataframe, output_path, model_name, downsample_2x)
