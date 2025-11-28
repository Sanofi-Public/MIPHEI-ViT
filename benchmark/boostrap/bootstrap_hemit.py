"""
Bootstrap analysis for cell type prediction using logistic regression outputs for HEMIT dataset.

This module provides functionality to perform bootstrap resampling on cell-level prediction results
to estimate the distribution of AUC and F1 scores for each marker. It loads prediction results,
applies a trained logistic regression model, and evaluates performance on test and validation sets
using parallelized computation.

Usage:
    python eval_hemit.py --checkpoint_dir <path_to_checkpoint_dir>  # or eval_hemit_hemit_pipeline
    python bootstrap_hemit.py --checkpoint_dir <path_to_checkpoint_dir>
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm


def run_boostrap_hemit_analysis(checkpoint_dir: str
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                           List[str]]:
    """
    Perform bootstrap analysis on cell type prediction results using logistic regression outputs.

    This function loads prediction results and ground truth labels, applies a trained logistic
    regression model to compute cell type probabilities, and then performs bootstrap resampling at
    the image level to estimate the distribution of AUC and F1 scores for each marker. The analysis
    is performed separately for test and validation datasets.
    Args:
        checkpoint_dir (str or Path): Path to the directory containing model checkpoints and
            prediction files.
    Returns:
        tuple:
            val_aucs (np.ndarray): Array of shape (n_bootstrap, n_markers) with bootstrapped AUCs
                for validation set.
            val_f1s (np.ndarray): Array of shape (n_bootstrap, n_markers) with bootstrapped F1
                scores for validation set.
            test_aucs (np.ndarray): Array of shape (n_bootstrap, n_markers) with bootstrapped AUCs
                for test set.
            test_f1s (np.ndarray): Array of shape (n_bootstrap, n_markers) with bootstrapped F1
                scores for test set.
            marker_names_target (list of str): List of marker names corresponding to the columns
                evaluated.
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

    test_aucs = []
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

        # Compute AUCs
        auc_markers = []
        f1_markers = []

        def compute_scores(marker_name):
            pred_col = f"{marker_name}_prob"
            true_col = f"{marker_name}_pos"
            auc = roc_auc_score(y_true=df_cell_sampled[true_col], y_score=df_cell_sampled[pred_col])
            f1 = f1_score(y_true=df_cell_sampled[true_col],
                          y_pred=(df_cell_sampled[pred_col] > 0.5).astype(int))
            return auc, f1

        results = Parallel(n_jobs=-1)(delayed(compute_scores)(marker_name)
                                      for marker_name in marker_names_target)
        auc_markers, f1_markers = map(list, zip(*results))

        test_aucs.append(np.hstack(auc_markers))
        test_f1s.append(np.hstack(f1_markers))

    test_aucs = np.vstack(test_aucs)
    test_f1s = np.vstack(test_f1s)

    val_aucs = []
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

        # Compute AUCs
        auc_markers = []
        f1_markers = []

        def compute_scores(marker_name):
            pred_col = f"{marker_name}_prob"
            true_col = f"{marker_name}_pos"
            auc = roc_auc_score(y_true=df_cell_sampled[true_col], y_score=df_cell_sampled[pred_col])
            f1 = f1_score(y_true=df_cell_sampled[true_col],
                          y_pred=(df_cell_sampled[pred_col] > 0.5).astype(int))
            return auc, f1

        results = Parallel(n_jobs=-1)(delayed(compute_scores)(marker_name)
                                      for marker_name in marker_names_target)
        auc_markers, f1_markers = map(list, zip(*results))

        val_aucs.append(np.hstack(auc_markers))
        val_f1s.append(np.hstack(f1_markers))

    val_aucs = np.vstack(val_aucs)
    val_f1s = np.vstack(val_f1s)

    return val_aucs, val_f1s, test_aucs, test_f1s, marker_names_target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    val_aucs, val_f1s, test_aucs, test_f1s, marker_names = run_boostrap_hemit_analysis(
        args.checkpoint_dir)
    with open(str(Path(args.checkpoint_dir) / "bootstrap_results_hemit.json"), "w") as f:
        json.dump({"val_aucs": val_aucs.tolist(), "val_f1s": val_f1s.tolist(),
                   "test_aucs": test_aucs.tolist(), "test_f1s": test_f1s.tolist(),
                   "marker_names": marker_names}, f)
