"""
Utility functions for cell-level evaluation.

Includes:
- XGBoost training and evaluation for multi-label classification.
- Correlation analysis and visualization for IMMUcan.
- Checkpoint adaptation for HEMIT models.
"""

from typing import Tuple
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from skimage.measure import regionprops_table
from skimage.color import rgb2hed
import pandas as pd
import torch
from torchmetrics import PearsonCorrCoef
from timm.layers import resample_patch_embed, resize_rel_pos_bias_table
from scipy.stats import pearsonr
import seaborn as sns
import tifffile
from concurrent.futures import as_completed

import matplotlib.pyplot as plt


class PixelPearsonCorrCoef(PearsonCorrCoef):
    """
    Pearson correlation for image tensors.
    Automatically flattens spatial dims.

    Accepts:
      (B, H, W)
      (B, C, H, W)   → requires channel index (or loop externally)

    Notes:
      - If (B, C, H, W) is provided, update() expects one channel at a time.
    """

    def __init__(self):
        # We treat this as 1 output regression
        super().__init__(num_outputs=1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds, target shapes:
          (B,   H, W)   → OK
          (B, C, H, W)  → user must index per channel before calling
        """
        # flatten spatial dims
        preds = preds.flatten() * 255  # avoid NaN due to very small values
        target = target.flatten() * 255
        # add channel dim → (N, 1)
        preds = preds.unsqueeze(-1)
        target = target.unsqueeze(-1)
        return super().update(preds.double(), target.double())


def get_celltype_metrics_df(results, nuclei_classes):
    rows = []
    for nuclei_class in nuclei_classes:
        row = [
            nuclei_class,
            results[f"{nuclei_class}_auc_logreg"].item(),
            results[f"{nuclei_class}_ap_logreg"].item(),
            results[f"{nuclei_class}_balanced_acc"].item(),
            results[f"{nuclei_class}_f1"].item(),
        ]
        rows.append(row)
    results_df = pd.DataFrame(
        rows, columns=["Marker", "ROC AUC", "AUPRC", "Balanced Accuracy", "F1 Score"])
    return results_df


def train_xgboost(train_cell_dataframe: pd.DataFrame, test_cell_dataframe: pd.DataFrame,
                  cell_metrics) -> Tuple[dict, pd.DataFrame]:
    """
    Train an XGBoost classifier for multi-label cell type classification and evaluates its \
    performance.

    Args:
        train_cell_dataframe (pd.DataFrame): DataFrame containing training cell-level data with
            marker prediction columns and marker columns.
        test_cell_dataframe (pd.DataFrame): DataFrame containing testing cell-level data with marker
            prediction columns and marker columns.
        cell_metrics (object): `CellMetrics` associated metric instance.
    Returns:
        model_dict (dict): Dictionary containing the trained model under the key "model" and the
            fitted scaler under the key "scaler".
        results_df (pd.DataFrame): DataFrame summarizing evaluation metrics (ROC AUC, Balanced
            Accuracy, F1 Score) for each marker/target.
    """
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
        scale_pos_weight=sum(y_train.ravel() == 0) / sum(y_train.ravel() == 1),  # Multi-label
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
    results_df = pd.DataFrame(
        results, columns=["Marker name", "ROC AUC", "Balanced Accuracy", "F1 Score"])
    model_dict = {"model": model, "scaler": scaler}
    return model_dict, results_df


def correlation_analysis(tile_sums: pd.DataFrame, marker_name: str, figpath: str,
                         colors_dict: dict) -> float:
    """
    Perform correlation analysis between target and predicted cell-type counts within tiles.

    Calculates the Pearson correlation coefficient between the target and predicted cell type counts
    (using logistic regression), creates a regression plot with the correlation value annotated,
    and saves the plot to the specified file path. Useful when the H&E and mIF WSIs are consecutive
    sections.
    Args:
        tile_sums (pd.DataFrame): DataFrame containing columns for true and predicted marker counts.
        marker_name (str): Name of the marker to analyze (used to select columns and for plot
            labeling).
        figpath (str): Path to save the generated plot image.
        colors_dict (dict): Dictionary mapping marker names to colors for plotting.
    Returns:
        float: Pearson correlation coefficient between true and predicted marker counts.
    """
    corr, _ = pearsonr(tile_sums[f"{marker_name}_count"], tile_sums[f"{marker_name}_pos_logreg"])
    formatted_corr = f"{corr:.3f}" if abs(corr) >= 0.01 else f"{corr:.2e}"

    sns.regplot(x=tile_sums[f"{marker_name}_count"], y=tile_sums[f"{marker_name}_pos_logreg"],
                line_kws={'color': 'black'}, color=colors_dict[marker_name], ci=None)
    plt.text(0.05, 0.95, f"Pearson r = {formatted_corr}", transform=plt.gca().transAxes,
             fontsize=20, verticalalignment='top',
             bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "gray"})

    # Better title without correlation
    plt.title(f"{marker_name}", fontsize=32)
    plt.xlabel('Target', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    plt.savefig(figpath, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return corr


def adapt_checkpoint_hemit(state_dict, model: torch.nn.Module) -> dict:
    """
    Adapt a checkpoint's state dictionary to match the architecture of the given model, \
    especially to handle differences in input size and changes in timm library versions.

    This function modifies the keys and values of the input `state_dict` to ensure compatibility
    with the provided `model`. It handles renaming of certain keys, skips unnecessary entries,
    and resizes weights or bias tables as needed for patch embedding and relative position bias.
    Used to adapt a checkpoint from HEMIT codebase.
    Args:
        state_dict (dict): The state dictionary loaded from a HEMIT checkpoint.
        model (torch.nn.Module): The model instance whose architecture the checkpoint should be
            adapted to.
    Returns:
        dict: A new state dictionary compatible with the provided model.
    """
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
        if any(n in k for n in ('relative_position_index', 'attn_mask')):
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
            if (v.shape != m.relative_position_bias_table.shape or
                    m.window_size[0] != m.window_size[1]):
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        new_state_dict[k] = v

    return new_state_dict


def build_pred_target_mapping(target_channel_names, dataset_name, marker_metadata_df):
    """
    Returns mapping so prediction + target can be aligned to intersection.

    Args:
        target_channel_names : list[str]
        dataset_name         : {"hemit", "orion", "pathocell"}
        marker_metadata_df   : pandas DataFrame
            columns: predicted_marker, hemit, orion, pathocell

    Returns:
        common_markers: list[str]
        pred_indices  : list[int]
        targ_indices  : list[int]
    """
    # column in metadata for this dataset
    col = dataset_name

    # predicted marker order (model output)
    pred_order = list(marker_metadata_df["predicted_marker"])

    # dataset column map (pred marker → equivalent name in dataset or NA)
    mapping = dict(zip(marker_metadata_df["predicted_marker"],
                       marker_metadata_df[col]))

    # reverse lookup target name → its index
    targ_index = {name: i for i, name in enumerate(target_channel_names)}

    common_markers = []
    pred_indices = []
    targ_indices = []

    for pred_i, pred_name in enumerate(pred_order):
        # name in target dataset (may equal or mapped)
        equiv = mapping[pred_name]

        if equiv == "NA":
            continue

        if equiv not in targ_index:
            continue

        # both exist!
        common_markers.append(pred_name)
        pred_indices.append(pred_i)
        targ_indices.append(targ_index[equiv])

    return common_markers, pred_indices, targ_indices


def extract_cell_features(he_image, label_mask):
    """
    he_image:   (H, W, 3) RGB H&E
    label_mask: (H, W)   int — unique cell ID per instance
    """

    # Convert to Hematoxylin channel
    hed = rgb2hed(he_image)
    H = hed[..., 0]

    props = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "extent",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "mean_intensity",
        "std_intensity",
        "max_intensity",
        "min_intensity",
    ]

    df = pd.DataFrame(
        regionprops_table(
            label_mask,
            intensity_image=H,
            properties=props
        )
    )
    return df


def save_tiff(out_array, path):
    tifffile.imwrite(path, out_array, compression="none")


def save_batch(out, executor, tile_names, output_dir):
    futures = []
    for i, name in enumerate(tile_names):
        arr = out[i]  # (16, W, H)
        path = str(Path(output_dir) / f"{name}.tiff")
        futures.append(executor.submit(save_tiff, arr, path))
    for f in as_completed(futures):
        f.result()  # ensure exceptions are raised
