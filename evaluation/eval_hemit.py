"""
Evaluate a model trained on ORION on HEMIT dataset to extract cell-level metrics and \
classification results.

This script loads a trained generator on ORION, runs inference on HEMIT data, computes
cell-level marker predictions, and evaluates classification performance (logistic regression,
XGBoost). Results are saved as CSV files.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import pyvips  # Avoid pyvips import error from src.dataset

from eval_utils import train_xgboost

sys.path.append("../")
from src.dataset import (
    get_width_height,
    NormalizationLayer,
    get_effective_width_height,
    TileImg2ImgSlideDataset
)
from src.generators import get_generator
from src.generators.hemit_models import resize_embed_hemit_statedict
from src.metrics import CellMetrics
from src.utils import validate_load_info, get_generator_state_dict


DATASET_CONFIG_PATH = "../configs/data/hemit.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint_dir')
    parser.add_argument('--inference_40x', action='store_true', help='Disable downsampling')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    inference_40x = args.inference_40x

    config_path = str(Path(checkpoint_dir) / "config.yaml")

    cfg = OmegaConf.load(config_path)
    cfg_data = OmegaConf.load(DATASET_CONFIG_PATH)
    for key in ["slide_dataframe_path", "train_dataframe_path", "val_dataframe_path",
                "test_dataframe_path", "channel_stats_path"]:
        if key in cfg_data.data:
            cfg.data[key] = cfg_data.data[key]

    slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)
    train_val_dataframe = pd.concat((
        pd.read_csv(cfg.data.train_dataframe_path),
        pd.read_csv(cfg.data.val_dataframe_path),
        ))
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)

    width, height = get_width_height(train_val_dataframe)
    width, height = get_effective_width_height(width, height, train=True)
    if inference_40x:
        inference_width = width
        inference_height = height
    else:
        inference_width = width // 2
        inference_height = height // 2

    channel_names = cfg.data.targ_channel_names
    nc_out = len(channel_names)
    nc_in = 3
    print("{} width / {} height".format(width, height))
    print("{} inputs channels / {} output channels".format(nc_in, nc_out))

    channel_stats_rgb = {"mean": cfg.data.normalization.mean,
                         "std": cfg.data.normalization.std}
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")
    preprocess_target_fn = NormalizationLayer(mode="if")

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, inference_width, nc_in, nc_out, cfg)
    use_safetensors = (Path(checkpoint_dir) / "model.safetensors").exists()
    if use_safetensors:
        from safetensors.torch import load_file
        checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
        state_dict = load_file(checkpoint_path, device="cpu")
        strict_load = False
        print("Loading checkpoint from safetensors")
    else:
        checkpoint_path = str(Path(checkpoint_dir) / "model.weights.ckpt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = get_generator_state_dict(state_dict)
        strict_load = True
        print("Loading checkpoint from ckpt")
    if hasattr(generator, "swinT"):
        state_dict = resize_embed_hemit_statedict(state_dict, generator)

    load_info = generator.load_state_dict(state_dict, strict=strict_load)
    if use_safetensors:
        validate_load_info(load_info)
    generator = generator.eval().cuda().half()

    train_val_dataset = TileImg2ImgSlideDataset(
            dataframe=train_val_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            spatial_augmentations=None, return_nuclei=True)
    test_dataset = TileImg2ImgSlideDataset(
            dataframe=test_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            spatial_augmentations=None, return_nuclei=True)

    num_workers = 6
    batch_size = 4
    device = "cpu"
    train_val_dataloader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=batch_size, pin_memory=device != "cpu",
        shuffle=False, drop_last=False, num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=device != "cpu",
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    cell_metrics = CellMetrics(slide_dataframe, marker_names=channel_names,
                               min_area=20).cuda()
    test_pix_metrics = MetricCollection(
            {
                "psnr_metric": PeakSignalNoiseRatio(data_range=(-0.9, 0.9)),
                "ssim_metric": StructuralSimilarityIndexMeasure(data_range=(-0.9, 0.9)),
            }).cuda()
    idxs_pix_metrics = [
        channel_names.index("Pan-CK"),
        channel_names.index("CD3e"),
        channel_names.index("Hoechst")
    ] # HEMIT: Pan-CK, CD3e, Hoechst order

    for batch in tqdm(train_val_dataloader):
        x = batch["image"].cuda()
        nuclei_masks = batch["nuclei"].cuda()
        slide_names = batch["slide_name"]

        with torch.inference_mode():
            x = torch.nn.functional.interpolate(x, (inference_width, inference_height),
                                                mode="bilinear")
            out = generator(x.half()).float()
            out = torch.nn.functional.interpolate(out, (width, height), mode="bilinear")

        cell_metrics.update(out, nuclei_masks, slide_names)

    for batch in tqdm(test_dataloader):
        x = batch["image"].cuda()
        y = batch["target"].cuda()
        nuclei_masks = batch["nuclei"].cuda()
        slide_names = batch["slide_name"]

        with torch.inference_mode():
            x = torch.nn.functional.interpolate(x, (inference_width, inference_height),
                                                mode="bilinear")
            out = generator(x.half()).float()
            out = torch.nn.functional.interpolate(out, (width, height), mode="bilinear")

        cell_metrics.update(out, nuclei_masks, slide_names)
        test_pix_metrics.update(out[:, idxs_pix_metrics].clip(-0.9, 0.9), y)

    # Tricks to adapt to HEMIT markers
    marker_names = ["Pan-CK", "CD3"]
    cell_metrics.marker_cols = marker_names
    cell_metrics.marker_cols = [f"{marker_name}_pos" for marker_name in marker_names]

    cell_dataframe = cell_metrics.get_dataframe_cell_pred_target()
    cell_metrics.reset()

    train_slide_names = list(pd.read_csv(cfg.data.train_dataframe_path)["in_slide_name"].unique())
    val_slide_names = list(pd.read_csv(cfg.data.val_dataframe_path)["in_slide_name"].unique())
    test_slide_names = list(test_dataframe["in_slide_name"].unique())

    # only 5% of cells from our pipeline
    train_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(
        train_slide_names)].sample(frac=0.05, random_state=42)
    val_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(val_slide_names)]
    test_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(test_slide_names)]

    # pixel level metrics
    test_pix_dicts = {k: [v.cpu().item()] for k, v in test_pix_metrics.compute().items()}
    results_pixel_df = pd.DataFrame(data=test_pix_dicts)
    results_pixel_df.to_csv(str(Path(checkpoint_path).parent / "hemit_results_pixel.csv"), index=False)

    # cell level classification
    # logistic regression
    results_test, logreg = cell_metrics.train_logistic_regression(
        train_cell_dataframe, test_cell_dataframe, return_metrics=True)
    results_test_df = pd.DataFrame(results_test, columns=["Marker", "ROC AUC", "Balanced Accuracy",
                                                          "F1 Score"])
    results_test_df["Set"] = "Test"

    results_val, _ = cell_metrics.train_logistic_regression(
        train_cell_dataframe, val_cell_dataframe, return_metrics=True)
    results_val_df = pd.DataFrame(results_val, columns=["Marker", "ROC AUC", "Balanced Accuracy",
                                                        "F1 Score"])
    results_val_df["Set"] = "Val"
    results_df = pd.concat((results_test_df, results_val_df), ignore_index=True)

    # xgboost
    _, results_test_df_xgboost = train_xgboost(
        train_cell_dataframe, test_cell_dataframe, cell_metrics)
    results_test_df_xgboost["Set"] = "Test"
    _, results_val_df_xgboost = train_xgboost(
        train_cell_dataframe, val_cell_dataframe, cell_metrics)
    results_val_df_xgboost["Set"] = "Val"
    results_df_xgboost = pd.concat(
        (results_test_df_xgboost, results_val_df_xgboost), ignore_index=True)

    results_df.to_csv(str(Path(checkpoint_path).parent / "hemit_results_logreg.csv"), index=False)
    results_df_xgboost.to_csv(
        str(Path(checkpoint_path).parent / "hemit_results_xgboost.csv"), index=False)

    val_test_test_cell_dataframe = pd.concat(
        (val_cell_dataframe, test_cell_dataframe), ignore_index=True)
    val_test_test_cell_dataframe.to_csv(
        str(Path(checkpoint_path).parent / "hemit_cell_dataframe.csv"), index=False)
    torch.save(logreg.state_dict(), str(Path(checkpoint_path).parent / "hemit_logreg.pth"))
