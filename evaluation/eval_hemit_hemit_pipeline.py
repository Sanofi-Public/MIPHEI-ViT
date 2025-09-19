"""
Evaluate a trained model on HEMIT dataset using HEMIT codebase on HEMIT dataset to extract \
cell-level metrics and classification results.

This script loads an HEMIT generator trained on HEMIT with HEMIT codebase, runs inference on HEMIT
data, computes cell-level marker predictions, and evaluates classification performance (logistic
regression, XGBoost). Results are saved as CSV files. Used to evaluate provided checkpoint from
HEMIT paper: https://drive.google.com/file/d/1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd/view?usp=sharing

HEMIT codebase: https://github.com/BianChang/Pix2pix_DualBranch
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import pyvips  # Avoid pyvips import error from src.dataset

from eval_utils import train_xgboost, adapt_checkpoint_hemit

sys.path.append("../")
from src.dataset import (
    get_width_height,
    NormalizationLayer,
    get_effective_width_height,
    TileImg2ImgSlideDataset,
)
from src.generators.hemit_models import get_generator_hemit
from src.metrics import CellMetrics


ORION_MARKERS = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin",
    "Ki67", "Pan-CK", "SMA"]
HEMIT_MARKERS = ["Pan-CK", "CD3", "DAPI"]
DATASET_CONFIG_PATH = "../configs/data/hemit.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint_dir')
    parser.add_argument('--trained_hemit', action='store_true',
                        help='true if the model is trained on HEMIT Dataset\
                              - used to match predicted marker names')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    trained_hemit = args.trained_hemit

    cfg = OmegaConf.load(DATASET_CONFIG_PATH)

    slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)
    train_val_dataframe = pd.concat((
        pd.read_csv(cfg.data.train_dataframe_path),
        pd.read_csv(cfg.data.val_dataframe_path)))
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)

    width, height = get_width_height(train_val_dataframe)
    width, height = get_effective_width_height(width, height, train=True)
    if trained_hemit:
        inference_width = width
        inference_height = height
    else:
        inference_width = width // 2
        inference_height = height // 2

    predicted_marker_names = HEMIT_MARKERS if trained_hemit else ORION_MARKERS
    nc_out = len(predicted_marker_names)
    nc_in = 3
    print("{} width / {} height".format(width, height))
    print("{} inputs channels / {} output channels".format(nc_in, nc_out))

    channel_stats_rgb = {"mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]}
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")
    preprocess_target_fn = NormalizationLayer(mode="if")

    torch.cuda.empty_cache()
    generator = get_generator_hemit(
        nc_in, nc_out, inference_width, ngf=64,
        netG="SwinTResnet", norm='batch', use_dropout=False,
        init_type='normal', init_gain=0.02, gpu_ids=[])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = adapt_checkpoint_hemit(state_dict, generator)
    generator.load_state_dict(state_dict)

    generator = generator.eval().cuda()

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
    cell_metrics = CellMetrics(slide_dataframe, marker_names=predicted_marker_names,
                               min_area=20).cuda()
    test_pix_metrics = MetricCollection(
            {
                "psnr_metric": PeakSignalNoiseRatio(data_range=(-0.9, 0.9)),
                "ssim_metric": StructuralSimilarityIndexMeasure(data_range=(-0.9, 0.9)),
            }).cuda()
    if not trained_hemit:
        idxs_pix_metrics = [
            ORION_MARKERS.index("Pan-CK"),
            ORION_MARKERS.index("CD3e"),
            ORION_MARKERS.index("Hoechst")
        ]
    else:
        idxs_pix_metrics = [0, 1, 2]

    for batch in tqdm(train_val_dataloader):
        x = batch["image"].cuda()
        nuclei_masks = batch["nuclei"].cuda()
        slide_names = batch["slide_name"]

        with torch.inference_mode():
            x = torch.nn.functional.interpolate(x, (inference_width, inference_height),
                                                mode="bilinear")
            out = generator(x)
            # scale output in [-0.9, 0.9] to match cell_metrics input
            out = (out + 1) / 2  # [-1, 1] -> [0, 1]
            out = out * 1.8 - 0.9  # [0, 1] -> [-0.9, 0.9]
            out = out.float()
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
            out = generator(x)
            # scale output in [-0.9, 0.9] to match cell_metrics input
            out = (out + 1) / 2  # [-1, 1] -> [0, 1]
            out = out * 1.8 - 0.9  # [0, 1] -> [-0.9, 0.9]
            out = out.float()
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

    train_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(
        train_slide_names)]
    if not trained_hemit:
        train_cell_dataframe = train_cell_dataframe.sample(frac=0.05, random_state=42)
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
    _, results_test_df_xgboost = train_xgboost(train_cell_dataframe, test_cell_dataframe,
                                               cell_metrics)
    results_test_df_xgboost["Set"] = "Test"
    _, results_val_df_xgboost = train_xgboost(train_cell_dataframe, val_cell_dataframe,
                                              cell_metrics)
    results_val_df_xgboost["Set"] = "Val"
    results_df_xgboost = pd.concat(
        (results_test_df_xgboost, results_val_df_xgboost), ignore_index=True)

    val_test_test_cell_dataframe = pd.concat(
        (val_cell_dataframe, test_cell_dataframe), ignore_index=True)
    val_test_test_cell_dataframe.to_csv(
        str(Path(checkpoint_path).parent / "hemit_cell_dataframe.csv"), index=False)
    results_df.to_csv(str(Path(checkpoint_path).parent / "hemit_results_logreg.csv"), index=False)
    torch.save(logreg.state_dict(), str(Path(checkpoint_path).parent / "hemit_logreg.pth"))
