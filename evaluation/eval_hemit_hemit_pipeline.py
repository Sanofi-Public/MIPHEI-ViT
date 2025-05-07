import pyvips
from omegaconf import OmegaConf
import json
import pandas as pd
from pathlib import Path
import torch
import argparse
from tqdm import tqdm
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from timm.layers import resample_patch_embed, resize_rel_pos_bias_table

import sys
sys.path.append("../")
from src.dataset import get_width_height, NormalizationLayer, get_effective_width_height,\
        TileImg2ImgSlideDataset, get_input_mean_std
from src.generators.hemit_models import get_generator_hemit
from src.metrics import CellMetrics


def adapt_checkpoint_hemit(state_dict, model):

    new_state_dict = {}
    for k, v in state_dict.items():
        if ".downsample.norm" in k or "downsample.reduction" in k:
            k_split = k.split(".")
            k_split[2] = str(int(k_split[2]) + 1)
            new_k = ".".join(k_split)
        elif 'relative_position_index' in k or 'attn_mask' in k:
            continue
        else:
            new_k = k
        new_state_dict[new_k] = v
    
    state_dict = new_state_dict
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'attn_mask')]):
                continue

        if 'swinT.patch_embed.proj.weight' in k:
            _, _, H, W = model.swinT.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        new_state_dict[k] = v

    return new_state_dict


def train_xgboost(train_cell_dataframe, test_cell_dataframe, cell_metrics):
    # Prepare the training and testing data
    X_train = train_cell_dataframe[cell_metrics.marker_pred_cols].values
    X_test = test_cell_dataframe[cell_metrics.marker_pred_cols].values
    y_train = train_cell_dataframe[cell_metrics.marker_cols].values
    y_test = test_cell_dataframe[cell_metrics.marker_cols].values

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize XGBClassifier with scale_pos_weight to handle class imbalance
    xgb_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=sum(y_train.ravel() == 0) / sum(y_train.ravel() == 1),  # Adjusted for multi-label
        random_state=42,
    )

    # Define OneVsRestClassifier with XGBClassifier
    model = OneVsRestClassifier(xgb_model)
    model.fit(X_train, y_train)

    # Predict probabilities and class labels
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Evaluate for each marker/target
    results = []
    for idx, marker_target in enumerate(cell_metrics.marker_cols):
        roc_auc = roc_auc_score(y_test[:, idx], y_proba[:, idx])
        balanced_acc = balanced_accuracy_score(y_test[:, idx], y_pred[:, idx])
        f1 = f1_score(y_test[:, idx], y_pred[:, idx])
        results.append((marker_target, roc_auc, balanced_acc, f1))

    # Display results in a DataFrame
    results_df = pd.DataFrame(results, columns=["Marker name", "ROC AUC", "Balanced Accuracy", "F1 Score"])
    model_dict = {"model": model, "scaler": scaler}
    return model_dict, results_df

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
    dataframe = pd.concat((
        pd.read_csv(cfg.data.train_dataframe_path),
        pd.read_csv(cfg.data.val_dataframe_path),
        pd.read_csv(cfg.data.test_dataframe_path)))

    width, height = get_width_height(dataframe)
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

    torch.cuda.empty_cache()
    generator = get_generator_hemit(
        nc_in, nc_out, inference_width, ngf=64,
        netG="SwinTResnet", norm='batch', use_dropout=False,
        init_type='normal', init_gain=0.02, gpu_ids=[])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = adapt_checkpoint_hemit(state_dict, generator)
    generator.load_state_dict(state_dict)

    generator = generator.eval().cuda()

    dataset = TileImg2ImgSlideDataset(
            dataframe=dataframe, preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=None, return_nuclei=True)

    num_workers = 6
    batch_size = 4
    device = "cpu"
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=device!="cpu",
        shuffle=False, drop_last=False, num_workers=num_workers
    )
    cell_metrics = CellMetrics(slide_dataframe, marker_names=predicted_marker_names,
                               min_area=20).cuda()

    for batch in tqdm(dataloader):
        x = batch["image"].cuda()
        nuclei_masks = batch["nuclei"].cuda()
        slide_names = batch["slide_name"]

        with torch.inference_mode():
            x = torch.nn.functional.interpolate(x, (inference_width, inference_height), mode="bilinear")
            out = generator(x)
            # scale output in [-0.9, 0.9] to match cell_metrics input
            out = (out  + 1) / 2 # [-1, 1] -> [0, 1]
            out = out * 1.8 - 0.9 # [0, 1] -> [-0.9, 0.9]
            out = out.float()
            out = torch.nn.functional.interpolate(out, (width, height), mode="bilinear")

        cell_metrics.update(out, nuclei_masks, slide_names)

    # Tricks to adapt to HEMIT markers
    marker_names = ["Pan-CK", "CD3"]
    cell_metrics.marker_cols = marker_names
    cell_metrics.marker_cols = [f"{marker_name}_pos" for marker_name in marker_names]

    cell_dataframe = cell_metrics.get_dataframe_cell_pred_target()
    cell_metrics.reset()

    train_slide_names = list(pd.read_csv(cfg.data.train_dataframe_path)["in_slide_name"].unique())
    val_slide_names = list(pd.read_csv(cfg.data.val_dataframe_path)["in_slide_name"].unique())
    test_slide_names = list(pd.read_csv(cfg.data.test_dataframe_path)["in_slide_name"].unique())

    train_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(
        train_slide_names)]
    if not trained_hemit:
        train_cell_dataframe = train_cell_dataframe.sample(frac=0.05, random_state=42)
    val_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(val_slide_names)]
    test_cell_dataframe = cell_dataframe[cell_dataframe["slide_name"].isin(test_slide_names)]

    # cell level classification
    # logistic regression
    results_test, _ = cell_metrics.train_logistic_regression(
        train_cell_dataframe, test_cell_dataframe, return_metrics=True)
    results_test_df = pd.DataFrame(results_test, columns=["Marker", "ROC AUC", "Balanced Accuracy", "F1 Score"])
    results_test_df["Set"] = "Test"

    results_val, _ = cell_metrics.train_logistic_regression(
        train_cell_dataframe, val_cell_dataframe, return_metrics=True)
    results_val_df = pd.DataFrame(results_val, columns=["Marker", "ROC AUC", "Balanced Accuracy", "F1 Score"])
    results_val_df["Set"] = "Val"
    results_df = pd.concat((results_test_df, results_val_df), ignore_index=True)

    # xgboost
    _, results_test_df_xgboost  = train_xgboost(train_cell_dataframe, test_cell_dataframe, cell_metrics)
    results_test_df_xgboost["Set"] = "Test"
    _, results_val_df_xgboost  = train_xgboost(train_cell_dataframe, val_cell_dataframe, cell_metrics)
    results_val_df_xgboost["Set"] = "Val"
    results_df_xgboost = pd.concat((results_test_df_xgboost, results_val_df_xgboost), ignore_index=True)

    results_df.to_csv(str(Path(checkpoint_path).parent / "hemit_results_logreg.csv"), index=False)
    results_df_xgboost.to_csv(str(Path(checkpoint_path).parent / "hemit_results_xgboost.csv"), index=False)
