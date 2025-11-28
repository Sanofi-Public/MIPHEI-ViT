"""
Bootstrap analysis for random model for cell type prediction using logistic regression outputs for \
ORION-CRC dataset.

This module performs bootstrap resampling on cell-level classification to estimate the
distribution of F1 scores for each marker for stratified random model. It loads prediction results,
applies a trained logistic regression model, and evaluates performance on test and validation sets
using parallelized computation.

Usage:
    python bootstrap_random_orion.py --checkpoint_dir <path_to_checkpoint_dir>
    # <path_to_checkpoint_dir> to have same targets of other models (can be any checkpoint
    # with evaluation run on ORION-CRC)
"""

from typing import List, Tuple

from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.utils import resample
from tqdm import tqdm

from bootstrap_orion import match_cell_and_tile


def run_boostrap_orion_analysis(checkpoint_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Perform bootstrap analysis on cell type classification for a stratified random model using \
    logistic regression on the ORION-CRC dataset.

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
    cfg = OmegaConf.load("../configs/data/orion.yaml")
    cell_prediction_path = str(Path(checkpoint_dir) / "cell_dataframe.csv")

    # Load Data
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    test_slide_names = test_dataframe["in_slide_name"].unique().tolist()

    slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)
    slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(test_slide_names)]
    df_cell = pd.read_csv(cell_prediction_path, engine="pyarrow")
    df_cell = df_cell[df_cell["slide_name"].isin(test_slide_names)]
    marker_names = [col.replace("_pos", "") for col in df_cell.columns if "_pos" in col]

    df_cell = match_cell_and_tile(df_cell, slide_dataframe, test_dataframe)

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=1000)
    f1s = []

    unique_images = df_cell['image_name'].dropna().unique()
    grouped_cells = df_cell.groupby('image_name')  # Group once, reuse in each iteration

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
                                         for marker_name in marker_names)

        f1s.append(np.hstack(f1_markers))

    f1s = np.vstack(f1s)

    return f1s, marker_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    f1s, marker_names = run_boostrap_orion_analysis(args.checkpoint_dir)
    with open(str(Path(args.checkpoint_dir).parent / "bootstrap_results_random_orion.json"),
              "w") as f:
        json.dump({"f1s": f1s.tolist(), "marker_names": marker_names}, f)
