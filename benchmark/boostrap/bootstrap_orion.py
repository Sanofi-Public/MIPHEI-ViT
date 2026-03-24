"""
Bootstrap analysis for cell type prediction using logistic regression outputs for ORION dataset.

This module provides functionality to perform bootstrap resampling on cell-level prediction results
to estimate the distribution of AUC and F1 scores for each marker. It loads prediction results,
applies a trained logistic regression model, and evaluates performance on test and validation sets
using parallelized computation.

Usage:
    python eval_orion.py --checkpoint_dir <path_to_checkpoint_dir>  # or eval_orion_hemit_pipeline
    python bootstrap_orion.py --checkpoint_dir <path_to_checkpoint_dir>
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
from shapely.geometry import Point, box
from shapely.strtree import STRtree
from sklearn.metrics import f1_score, average_precision_score
from sklearn.utils import resample
from tqdm import tqdm


def get_tile_infos(tile_name: str) -> pd.Series:
    """Extract the tile information from tile names in pandas Series."""
    tile_name_split = tile_name.split("_")[-5:]
    tile_name_split[-1] = tile_name_split[-1].split(".")[0]
    return pd.Series(list(map(int, tile_name_split)))


def match_cell_and_tile(df_cell: pd.DataFrame, slide_dataframe: pd.DataFrame,
                        test_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Match each nucleus (cell) to its corresponding tile based on spatial location.

    This utility function assigns the correct tile image name to each nucleus (cell) in the
    provided cell DataFrame, based on the spatial coordinates of the nuclei and the tile boundaries
    defined in the test DataFrame. It is used to do tile sampling during bootstrapping.

    Args:
        df_cell (pd.DataFrame): DataFrame containing nuclei information, including coordinates and
            slide names. Obtained from evaluation step.
        slide_dataframe (pd.DataFrame): DataFrame containing slide-level metadata.
        test_dataframe (pd.DataFrame): DataFrame containing tile information, including image paths
            and spatial boundaries.

    Returns:
        pd.DataFrame: The input df_cell DataFrame with an additional 'image_name' column indicating
            the matched tile for each nucleus.

    Raises:
        AssertionError: If any nucleus could not be matched to a tile (i.e., if any 'image_name'
            remains NaN).
    """
    tile_info_df = test_dataframe["image_path"].apply(get_tile_infos)
    # Rename the columns to match the structure you described
    tile_info_df.columns = ["x", "y", "level", "tile_size_x", "tile_size_y"]

    # Join the new columns with the original DataFrame
    test_dataframe = test_dataframe.join(tile_info_df)

    del tile_info_df; gc.collect()
    test_dataframe["image_name"] = test_dataframe["image_path"].apply(lambda x: Path(x).stem)
    df_cell['image_name'] = None

    for slide_name in df_cell["slide_name"].unique():
        test_dataframe_slide = test_dataframe[test_dataframe["in_slide_name"] == slide_name]
        df_cell_slide = df_cell[df_cell["slide_name"] == slide_name]

        # Build tile polygons and names
        tiles = test_dataframe_slide.apply(
            lambda row: box(row['x'], row['y'], row['x'] + row['tile_size_x'],
                            row['y'] + row['tile_size_y']),
            axis=1
        ).tolist()
        tile_names = test_dataframe_slide['image_name'].tolist()

        # Build STRtree on cell centroids
        df_cell_slide = df_cell_slide.copy()
        df_cell_slide['geometry'] = df_cell_slide.apply(
            lambda row: Point(row['x'], row['y']), axis=1)
        cell_geometries = df_cell_slide['geometry'].tolist()
        tree = STRtree(cell_geometries)
        geom_to_index = dict(zip(cell_geometries, df_cell_slide.index))

        # Assign image_name to cells inside any tile
        for tile_geom, tile_name in zip(tiles, tile_names):
            for idx_cell in tree.query(tile_geom):
                pt = tree.geometries[idx_cell]
                if tile_geom.contains(pt):
                    idx = geom_to_index[pt]
                    df_cell.at[idx, 'image_name'] = tile_name

        del test_dataframe_slide, cell_geometries; gc.collect()
    assert len(df_cell[df_cell['image_name'].isna()]) == 0
    return df_cell


def run_boostrap_orion_analysis(checkpoint_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    cfg = OmegaConf.load("../configs/data/orion.yaml")
    cell_prediction_path = str(Path(checkpoint_dir) / "orion_cell_dataframe_logreg.parquet")

    # Load Data
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    test_slide_names = test_dataframe["in_slide_name"].unique().tolist()

    slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)
    slide_dataframe = slide_dataframe[slide_dataframe["in_slide_name"].isin(test_slide_names)]
    df_cell = pd.read_parquet(cell_prediction_path)
    df_target = pd.read_parquet(cfg.data.nuclei_dataframe_path)[["slide_name", "label", "x", "y"]]
    df_cell = df_cell.merge(
        df_target, left_on=["slide_name", "cell_id"], right_on=["slide_name", "label"], how="left")

    df_cell = df_cell[df_cell["slide_name"].isin(test_slide_names)]
    marker_names_target = [col.replace("_pos", "") for col in df_cell.columns if "_pos" in col]
    marker_names_pred = [col.replace("_pred", "") for col in df_cell.columns if "_pred" in col]

    df_cell = match_cell_and_tile(df_cell, slide_dataframe, test_dataframe)

    # Load logistic regression model and predict cell types
    linear_logreg = torch.nn.Linear(len(marker_names_pred), len(marker_names_target))
    state_dict_logreg = torch.load(str(Path(checkpoint_dir) / "orion_logreg.pth"), map_location="cpu")
    linear_logreg.load_state_dict(state_dict_logreg)
    linear_logreg.eval()

    pred = df_cell[[f"{marker_name}_pred" for marker_name in marker_names_pred]].to_numpy().astype(
        np.float32)
    with torch.inference_mode():
        probs = torch.sigmoid(linear_logreg(torch.from_numpy(pred))).numpy()
    del pred; gc.collect()
    df_cell[[f"{marker_name}_prob" for marker_name in marker_names_target]] = probs

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=1000)
    aucs = []
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

        # Compute AUCs
        auc_markers = []
        f1_markers = []

        def compute_scores(marker_name):
            pred_col = f"{marker_name}_prob"
            true_col = f"{marker_name}_pos"
            ap_auc = average_precision_score(y_true=df_cell_sampled[true_col], y_score=df_cell_sampled[pred_col])
            f1 = f1_score(y_true=df_cell_sampled[true_col],
                          y_pred=(df_cell_sampled[pred_col] > 0.5).astype(int))
            return ap_auc, f1

        results = Parallel(n_jobs=-1)(delayed(compute_scores)(marker_name)
                                      for marker_name in marker_names_target)
        auc_markers, f1_markers = map(list, zip(*results))

        aucs.append(np.hstack(auc_markers))
        f1s.append(np.hstack(f1_markers))

    aucs = np.vstack(aucs)
    f1s = np.vstack(f1s)

    return aucs, f1s, marker_names_target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    aucs, f1s, marker_names = run_boostrap_orion_analysis(args.checkpoint_dir)
    with open(str(Path(args.checkpoint_dir) / "bootstrap_results.json"), "w") as f:
        json.dump({"ap_aucs": aucs.tolist(), "f1s": f1s.tolist(), "marker_names": marker_names}, f)
