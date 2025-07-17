"""
Bootstrap analysis for random model for cell type prediction using logistic regression outputs for \
HEMIT dataset.

This module performs bootstrap resampling on cell-level classification to estimate the
distribution of F1 scores for each marker for stratified random model. It loads prediction results,
applies a trained logistic regression model, and evaluates performance on test and validation sets
using parallelized computation.

Usage:
    python bootstrap_random_hemit.py --checkpoint_dir <path_to_checkpoint_dir>
    # <path_to_checkpoint_dir> to have same targets of other models (can be any checkpoint
    # with evaluation run on HEMIT)
"""

from typing import List, Tuple

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.utils import resample
from tqdm import tqdm


def run_boostrap_hemit_analysis(checkpoint_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform bootstrap analysis on cell type classification for a stratified random model using \
    logistic regression on the HEMIT dataset.

    This function loads cell prediction data from a checkpoint to ensure the same ground truth
    nuclei are used for evaluation as in other models. This is necessary because the ground truth
    CSV may contain all nuclei in the WSI, but only a subset (those in validation/test patches) are
    evaluated by the models. The function applies a trained logistic regression model to compute
    cell type probabilities, then performs bootstrap resampling at the image level to estimate the
    distribution of F1 scores for each marker. The process is repeated for both test and validation
    datasets.
    Args:
        checkpoint_dir (str): Path to a directory containing the cell prediction CSV and
            trained logistic regression checkpoint. This ensures the same nuclei are evaluated as
            in other models, since only nuclei present in the prediction CSV are considered.
    Returns:
        val_f1s (np.ndarray): Array of shape (1000, n_markers) containing bootstrapped F1 scores
            for the validation set.
        test_f1s (np.ndarray): Array of shape (1000, n_markers) containing bootstrapped F1 scores
            for the test set.
        marker_names_target (list of str): List of marker names corresponding to the F1 score
            columns.
    """
    cfg = OmegaConf.load("../configs/data/hemit.yaml")
    cell_prediction_path = str(Path(checkpoint_dir) / "hemit_cell_dataframe.csv")

    # Load Data
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    test_image_names = test_dataframe["in_slide_name"].unique().tolist()
    val_dataframe = pd.read_csv(cfg.data.val_dataframe_path)
    val_image_names = val_dataframe["in_slide_name"].unique().tolist()

    df_cell = pd.read_csv(cell_prediction_path, engine="pyarrow")
    df_cell.rename(columns={"slide_name": "image_name"}, inplace=True)
    marker_names_pred = [col.replace("_pred", "") for col in df_cell.columns if "_pred" in col]
    marker_names_target = [col.replace("_pos", "") for col in df_cell.columns if "_pos" in col]

    # Load logistic regression model and predict cell types
    n_markers_in = len(marker_names_pred)
    linear_logreg = torch.nn.Linear(n_markers_in, len(marker_names_target))
    state_dict_logreg = torch.load(str(Path(checkpoint_dir) / "hemit_logreg.pth"),
                                   map_location="cpu")
    linear_logreg.load_state_dict(state_dict_logreg)
    linear_logreg.eval()

    pred = df_cell[[f"{marker_name}_pred" for marker_name in marker_names_pred]].to_numpy().astype(
        np.float32)
    with torch.inference_mode():
        probs = torch.sigmoid(linear_logreg(torch.from_numpy(pred))).numpy()
    del pred; gc.collect()
    df_cell[[f"{marker_name}_prob" for marker_name in marker_names_target]] = probs

    df_cell_test = df_cell[df_cell["image_name"].isin(test_image_names)]
    df_cell_val = df_cell[df_cell["image_name"].isin(val_image_names)]

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=1000)

    test_f1s = []
    unique_images = df_cell_test['image_name'].dropna().unique()
    grouped_cells = df_cell_test.groupby('image_name')  # Group once, reuse in each iteration

    for idx_set in tqdm(range(1000)):
        # Sample image_names with replacement
        sampled_images = resample(unique_images, replace=True, n_samples=len(unique_images),
                                  random_state=seeds[idx_set])

        # Reindex sampled image groups efficiently
        sampled_cells = [grouped_cells.get_group(img) for img in sampled_images]
        df_cell_sampled = pd.concat(sampled_cells, ignore_index=True)

        f1_markers = []

        def compute_scores(marker_name):
            true_col = f"{marker_name}_pos"
            target_marker = df_cell_sampled[true_col].values
            marker_prob = target_marker.mean()
            f1_scores = []
            for _ in range(10):
                random_pred = np.random.uniform(size=len(target_marker)) < marker_prob
                f1_scores.append(f1_score(y_true=target_marker, y_pred=random_pred))
            f1_scores = np.asarray(f1_scores)
            return f1_scores.mean()

        f1_markers = Parallel(n_jobs=-1)(delayed(compute_scores)(marker_name)
                                         for marker_name in marker_names_target)
        test_f1s.append(np.hstack(f1_markers))

    test_f1s = np.vstack(test_f1s)

    val_f1s = []
    unique_images = df_cell_val['image_name'].dropna().unique()
    grouped_cells = df_cell_val.groupby('image_name')  # Group once, reuse in each iteration

    for idx_set in tqdm(range(1000)):
        # Sample image_names with replacement
        sampled_images = resample(unique_images, replace=True, n_samples=len(unique_images),
                                  random_state=seeds[idx_set])

        # Reindex sampled image groups efficiently
        sampled_cells = [grouped_cells.get_group(img) for img in sampled_images]
        df_cell_sampled = pd.concat(sampled_cells, ignore_index=True)

        f1_markers = []

        def compute_scores(marker_name):
            true_col = f"{marker_name}_pos"
            target_marker = df_cell_sampled[true_col].values
            marker_prob = target_marker.mean()
            f1_scores = []
            for _ in range(10):
                random_pred = np.random.uniform(size=len(target_marker)) < marker_prob
                f1_scores.append(f1_score(y_true=target_marker, y_pred=random_pred))
            f1_scores = np.asarray(f1_scores)
            return f1_scores.mean()

        f1_markers = Parallel(n_jobs=-1)(delayed(compute_scores)(marker_name)
                                         for marker_name in marker_names_target)
        val_f1s.append(np.hstack(f1_markers))

    val_f1s = np.vstack(val_f1s)

    return val_f1s, test_f1s, marker_names_target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    val_f1s, test_f1s, marker_names = run_boostrap_hemit_analysis(args.checkpoint_dir)
    with open(str(Path(args.checkpoint_dir).parent / "bootstrap_results_random_hemit.json"),
              "w") as f:
        json.dump({"val_f1s": val_f1s.tolist(),
                   "test_f1s": test_f1s.tolist(),
                   "marker_names": marker_names}, f)
