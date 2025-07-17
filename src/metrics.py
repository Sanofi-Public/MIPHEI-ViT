"""Implementations of cell metrics for evaluating cell-level predictions."""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from torchmetrics import Metric


class CellMetrics(Metric):
    """
    Compute cell-level metrics for multi-marker image analysis.

    This metric class aggregates predictions and ground truth at the cell level from mIF image
    predictions, computes various classification metrics (AUC, balanced accuracy, F1), and supports
    logistic regression calibration for marker predictions. It is designed for use in multi-marker
    immunofluorescence or similar imaging experiments, where each cell may be positive for multiple
    markers.

    Args:
        slide_dataframe (pd.DataFrame): DataFrame containing slide metadata, including
            'in_slide_name' and 'nuclei_csv_path' columns.
        marker_names (List[str]): List of marker names to include in the analysis. Nuclei CSVs
            should contain the columns marker_name with _pos (e.g. "CD3_pos" for CD3).
        min_area (int, optional): Minimum area threshold for a cell to be included in
            the metrics computation. Defaults to 20.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Attributes:
        marker_names (List[str]): Filtered marker names (excluding "Hoechst" and "Dapi").
        marker_idxs (List[int]): Indices of the included markers.
        intensity_prop_col_names (List[str]): Column names for mean intensity per marker.
        slide_names (List[str]): List of slide names.
        csv_path_dict (Dict[str, str]): Mapping from slide name to nuclei CSV path.
        marker_cols (List[str]): Column names for ground truth marker positivity.
        marker_pred_cols (List[str]): Column names for predicted marker positivity.
        min_area (int): Minimum area threshold for cell inclusion.

    Methods:
        update(preds, nuclei_masks, slide_names):
            Updates internal state with new batch predictions and nuclei masks.

        compute(logreg_layer=None, return_dataframe=False):
            Computes metrics (AUC, balanced accuracy, F1) for each marker and overall.
            Optionally calibrates predictions using logistic regression.

        get_dataframe_cell_pred():
            Returns a DataFrame of per-cell predictions aggregated across all slides.

        get_dataframe_cell_target(slide_names=None):
            Returns a DataFrame of per-cell ground truth labels for specified slides.

        get_dataframe_cell_pred_target():
            Returns a merged DataFrame of per-cell predictions and ground truth labels.

        train_logistic_regression(train_dataframe, test_dataframe=None, return_metrics=True):
            Trains a logistic regression model (one-vs-rest) for marker prediction calibration.
            Returns metrics and a PyTorch Linear layer with calibrated weights.
    """

    def __init__(self, slide_dataframe: pd.DataFrame, marker_names: List[str], min_area: int = 20,
                 **kwargs):
        super().__init__(dist_sync_on_step=False, compute_on_cpu=True, **kwargs)
        excluded_markers = ["Hoechst", "Dapi"]
        filtered_names = [(i, name) for i, name in enumerate(marker_names)
                          if name not in excluded_markers]
        self.marker_names = [name for _, name in filtered_names]
        self.marker_idxs = [idx for idx, _ in filtered_names]
        self.intensity_prop_col_names = [f"mean_intensity-{idx}"
                                         for idx in range(len(self.marker_names))]
        self.slide_names = slide_dataframe["in_slide_name"].tolist()
        self.csv_path_dict = {}
        self.marker_cols = [f"{marker_name}_pos" for marker_name in self.marker_names]
        self.marker_pred_cols = [f"{marker_name}_pred" for marker_name in self.marker_names]
        self.min_area = min_area
        for _, row in slide_dataframe.iterrows():
            slide_name = row["in_slide_name"]
            csv_path = row["nuclei_csv_path"]
            self.csv_path_dict[slide_name] = csv_path
            self.add_state(f"{slide_name}_cell_id", default=[], dist_reduce_fx="cat")
            self.add_state(f"{slide_name}_sum", default=[], dist_reduce_fx="cat")
            self.add_state(f"{slide_name}_area", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, nuclei_masks: torch.Tensor, slide_names: List[str]
               ) -> None:
        """
        Update the metric state with new mIF predictions and nuclei masks for a batch of slides.

        This method processes the predicted marker values and nuclei segmentation masks for each
        image in the batch. For each slide, it aggregates per-nucleus marker predictions, computes
        region-wise sums, and updates internal storage for cell IDs, marker sums, and region areas.
        Args:
            preds (torch.Tensor): Predicted marker values of shape [batch_size, num_markers, H, W]
                in range [-1., 1.].
            nuclei_masks (torch.Tensor): Nuclei segmentation masks of shape [batch_size, H, W],
                where each pixel contains a label corresponding to a nucleus or background.
            slide_names (List[str]): List of slide names corresponding to each item in the batch.
        Returns:
            None
        """
        preds = torch.clip(preds[:, self.marker_idxs], -0.9, 0.9).float()
        preds = (preds + 0.9) / 1.8  # Range [-1, 1] -> [0, 1]

        nuclei_masks = torch.unsqueeze(nuclei_masks, dim=1).float()

        num_channels = len(self.marker_names)
        # Loop over batch because same cell ids can appear from different WSIs
        for idx_batch in range(len(nuclei_masks)):
            nuclei_b = nuclei_masks[idx_batch, 0]  # Shape: [H, W]
            pred_b = preds[idx_batch]
            slide_name = slide_names[idx_batch]

            # Create a binary mask for non-background pixels
            nuclei_binary = nuclei_b > 0

            # Apply the binary mask to nuclei
            nuclei_flat = nuclei_b[nuclei_binary]  # Shape: [num_valid_pixels]
            if nuclei_flat.numel() == 0:  # No valid regions
                continue

            # Get unique labels and their indices
            unique_labels, inverse_indices = torch.unique(nuclei_flat, return_inverse=True)

            # Extract predicted marker values for all non-background pixels,
            # flattening to shape [num_valid_pixels, num_channels]
            pred_flat = pred_b.permute(1, 2, 0)[nuclei_binary]

            # For each unique cell label, sum the predicted marker values across all its pixels.
            # pred_sums: [num_cells, num_channels]
            # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html
            pred_sums = torch.zeros(
                (unique_labels.shape[0], num_channels), dtype=preds.dtype,
                device=preds.device).scatter_add_(
                    0, inverse_indices.unsqueeze(1).expand(-1, num_channels), pred_flat)

            # For each unique cell label, count the number of pixels belonging to that cell.
            # region_counts: [num_cells]
            region_counts = torch.zeros(
                unique_labels.shape[0], dtype=torch.float32,
                device=nuclei_masks.device).scatter_add_(
                    0, inverse_indices, torch.ones_like(nuclei_flat, dtype=torch.float32))

            # We do not directly average here because nuclei can appear on multiple tiles and
            # be seen several times.
            unique_labels = unique_labels.to(torch.uint32).cpu()
            region_counts = torch.unsqueeze(region_counts.to(torch.uint16).cpu(), dim=-1)
            # To reduce RAM usage, we compress to uint32 and multiply by 255 to avoid
            # excessive rounding.
            pred_sums = (pred_sums * 255).to(torch.uint32).cpu()

            getattr(self, f"{slide_name}_cell_id").append(unique_labels)
            getattr(self, f"{slide_name}_sum").append(pred_sums)
            getattr(self, f"{slide_name}_area").append(region_counts)

    def compute(self, logreg_layer: torch.nn.Linear = None, return_dataframe: bool = False):
        """
        Compute evaluation metrics for marker predictions from metric states.

        This method calculates metrics such as AUC, balanced accuracy, and F1 score for each marker,
        as well as their averages across all markers. If a logistic regression layer is not
        provided, it will be trained using the current predictions. The method can also return the
        underlying dataframe used for computation if requested.

        Args:
            logreg_layer (torch.nn.Module, optional): Pre-trained logistic regression layer to use
                for evaluation. If None, a new logistic regression layer will be trained.
                Defaults to None.
            return_dataframe (bool, optional): Whether to return the dataframe used for metric
                computation along with the metrics dictionary. Defaults to False.

        Returns:
            dict: A dictionary containing per-marker and averaged metrics, as well as the
                state_dict of the logistic regression layer.
            pandas.DataFrame (optional): The dataframe containing predictions and targets, returned
                if `return_dataframe` is True.
        """
        dataframe = self.get_dataframe_cell_pred_target()

        metrics = {}
        metrics["auc"] = 0
        metrics["auc_logreg"] = 0
        metrics["balanced_acc"] = 0
        metrics["f1"] = 0
        train_logreg = logreg_layer is None
        if train_logreg:
            logreg_layer = self.train_logistic_regression(dataframe, return_metrics=False)

        preds = dataframe[self.marker_pred_cols].to_numpy()
        targets = dataframe[self.marker_cols].to_numpy()
        with torch.inference_mode():
            logreg_device = next(logreg_layer.parameters()).device
            with torch.amp.autocast(str(self.device), dtype=self.dtype):
                logreg_probs = torch.sigmoid(logreg_layer(
                    torch.from_numpy(preds).to(logreg_device)))
                logreg_preds = logreg_probs > 0.5
        logreg_probs = logreg_probs.cpu().numpy()
        logreg_preds = logreg_preds.cpu().numpy()

        for idx_marker, marker_col in enumerate(self.marker_cols):
            targets_marker = targets[..., idx_marker]
            preds_marker = preds[..., idx_marker]
            logreg_probs_marker = logreg_probs[..., idx_marker]
            logreg_preds_marker = logreg_preds[..., idx_marker]
            if (len(targets) == 0) or (len(np.unique(targets)) == 1):
                continue
            auc = torch.tensor(
                roc_auc_score(y_true=targets_marker, y_score=preds_marker), dtype=torch.float32)
            auc_logreg = torch.tensor(
                roc_auc_score(y_true=targets_marker, y_score=logreg_probs_marker),
                dtype=torch.float32)
            balanced_acc = torch.tensor(
                balanced_accuracy_score(y_true=targets_marker, y_pred=logreg_preds_marker),
                dtype=torch.float32)
            f1 = torch.tensor(f1_score(y_true=targets_marker, y_pred=logreg_preds_marker),
                              dtype=torch.float32)

            metrics[f"{marker_col}_auc"] = auc
            metrics[f"{marker_col}_auc_logreg"] = auc_logreg
            metrics[f"{marker_col}_balanced_acc"] = balanced_acc
            metrics[f"{marker_col}_f1"] = f1
            metrics["auc"] += auc
            metrics["auc_logreg"] += auc_logreg
            metrics["balanced_acc"] += balanced_acc
            metrics["f1"] += f1

        metrics["auc"] /= len(self.marker_names)
        metrics["auc_logreg"] /= len(self.marker_names)
        metrics["balanced_acc"] /= len(self.marker_names)
        metrics["f1"] /= len(self.marker_names)
        metrics["state_dict"] = logreg_layer.state_dict()
        self.reset()
        if return_dataframe:
            return metrics, dataframe
        else:
            return metrics

    def get_dataframe_cell_pred(self) -> pd.DataFrame:
        """
        Aggregate and average cell-level predicted expressions for each slide into a single \
        DataFrame.

        For each slide in `self.slide_names`, this method:
            - Retrieves cell IDs, prediction sums, and cell areas from `self.metric_state`.
            - Aggregates metrics by cell ID, summing values for duplicate IDs.
            - Filters out cells with area less than `self.min_area`.
            - Averages prediction sums by cell area.
            - Adds a column indicating the slide name.
        The results for all slides are concatenated into a single DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing cell-level predictions, averaged by area,
                with columns for cell ID, marker predictions, area, and slide name.
        """
        dataframe = []
        for slide_name in self.slide_names:
            dataframe_slide = pd.DataFrame()
            cell_ids = self.metric_state[f"{slide_name}_cell_id"]
            if len(cell_ids) == 0:
                continue
            cell_ids = torch.hstack(cell_ids).numpy()  # dim_zero_cat
            sums = torch.vstack(self.metric_state[f"{slide_name}_sum"]).numpy()
            areas = torch.vstack(self.metric_state[f"{slide_name}_area"]).numpy()
            dataframe_slide["cell_id"] = np.uint64(cell_ids)
            dataframe_slide[self.marker_pred_cols] = sums
            dataframe_slide["area"] = areas
            columns_groupby = list(dataframe_slide.columns)
            columns_groupby.remove('cell_id')
            dataframe_slide = dataframe_slide.groupby('cell_id')[
                columns_groupby].sum().reset_index(drop=False)
            dataframe_slide = dataframe_slide[dataframe_slide['area'] > self.min_area]
            dataframe_slide[self.marker_pred_cols] = dataframe_slide[
                self.marker_pred_cols].astype(np.float32).div(
                    dataframe_slide["area"], axis=0)
            dataframe_slide["slide_name"] = pd.Categorical([slide_name] * len(dataframe_slide))
            dataframe.append(dataframe_slide)
        dataframe = pd.concat(dataframe, ignore_index=True)
        return dataframe

    def get_dataframe_cell_target(self, slide_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and concatenate target data from CSV files for specified slides.

        Reads the 'label' (cell id) column and marker columns from each slide's CSV file,
        adds a 'slide_name' column, and concatenates the results into a single DataFrame.
        Args:
            slide_names (list of str, optional): List of slide names to process.
                If None, uses self.slide_names.
        Returns:
            pandas.DataFrame: Concatenated DataFrame containing cell-level target data for the
                specified slides, including 'label' (cell id), marker columns, and 'slide_name'.
        """
        usecols = ["label"] + self.marker_cols

        dataframe_target = []
        if slide_names is None:
            slide_names = self.slide_names
        for slide_name in slide_names:
            csv_path = self.csv_path_dict[slide_name]
            dataframe_slide_target = pd.read_csv(csv_path, engine="pyarrow", usecols=usecols)
            dataframe_slide_target["slide_name"] = pd.Categorical(
                [slide_name] * len(dataframe_slide_target))
            dataframe_target.append(dataframe_slide_target)

        dataframe_target = pd.concat(dataframe_target, ignore_index=True)
        return dataframe_target

    def get_dataframe_cell_pred_target(self) -> pd.DataFrame:
        """
        Merge predicted cell data with target cell data for evaluation.

        Retrieves the predicted cell dataframe and the target cell dataframe, merges them on slide
        name and cell ID, filters out rows without a matching label (cell was not seen during
        evaluation), and processes marker columns to ensure boolean type.
        Returns:
            pandas.DataFrame: A dataframe containing merged prediction and target information for
                each cell, with marker columns as boolean values.
        """
        dataframe = self.get_dataframe_cell_pred()
        dataframe_target = self.get_dataframe_cell_target(
            slide_names=dataframe["slide_name"].unique())

        dataframe = dataframe.merge(
            dataframe_target, left_on=["slide_name", "cell_id"],
            right_on=["slide_name", "label"], how="left")

        dataframe = dataframe[~dataframe["label"].isna()]
        dataframe = dataframe.drop(columns=["area", "label"])
        dataframe[self.marker_cols].astype(bool).fillna(False, inplace=True)
        dataframe[self.marker_cols] = dataframe[self.marker_cols].astype(bool)
        return dataframe

    def train_logistic_regression(self, train_dataframe: pd.DataFrame,
                                  test_dataframe: Optional[pd.DataFrame] = None,
                                  return_metrics: bool = True):
        """
        Train logistic regression model for multi-label cell type classification from cell \
        expressions.

        This method fits a set of logistic regression classifiers (one per marker/target) using the
        provided training dataframe (containing training mean cell expressions and target cell
        types). Features are standardized before training. If a test dataframe is provided, metrics
        are computed on the test set; otherwise, metrics are computed on the training set. The
        method also converts the trained scikit-learn models into a PyTorch Linear layer with
        adjusted weights and biases to account for feature standardization.
        Args:
            train_dataframe (pd.DataFrame): DataFrame containing training data with feature columns
                (_pred suffix) and marker columns (_pos suffix).
            test_dataframe (Optional[pd.DataFrame], optional): DataFrame containing test data.
                If None, uses training data as test set. Defaults to None.
            return_metrics (bool, optional): Whether to return evaluation metrics along with the
                PyTorch layer. Defaults to True.
        Returns:
            If `return_metrics` is True:
                results: List of tuples for each marker/target containing:
                    (marker_target, roc_auc, balanced_accuracy, f1_score)
            logreg_layer (torch.nn.Linear): PyTorch Linear layer with adjusted weights and biases.
        """
        # Prepare training data
        X_train = train_dataframe[self.marker_pred_cols].values

        # If test_dataframe is None, use X_train as X_test
        if test_dataframe is None:
            X_test = X_train
            y_test = train_dataframe[self.marker_cols].values
        else:
            X_test = test_dataframe[self.marker_pred_cols].values
            y_test = test_dataframe[self.marker_cols].values

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Convert labels to multi-label format
        y_train = train_dataframe[self.marker_cols].values

        # Define OneVsRestClassifier with LogisticRegression
        model = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
        model.fit(X_train, y_train)
        if return_metrics:
            X_test = scaler.transform(X_test)
            # Predict probabilities and class labels
            y_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # Evaluate for each marker/target
            results = []
            for idx, marker_target in enumerate(self.marker_cols):
                roc_auc = roc_auc_score(y_test[:, idx], y_proba[:, idx])
                balanced_acc = balanced_accuracy_score(y_test[:, idx], y_pred[:, idx])
                f1 = f1_score(y_test[:, idx], y_pred[:, idx])
                results.append((marker_target, roc_auc, balanced_acc, f1))

        # Compute adjusted weights and bias
        means = scaler.mean_
        stds = scaler.scale_
        weights = np.vstack([est.coef_.flatten() if hasattr(est, "coef_")
                             else np.zeros(len(self.marker_cols))
                             for est in model.estimators_])  # avoid constant model error
        bias = np.hstack([est.intercept_.flatten() if hasattr(est, "intercept_")
                          else 0. for est in model.estimators_])
        # Adjust weights and bias for standardized input
        adjusted_weights = weights / stds
        adjusted_bias = bias - np.sum((weights * means / stds), axis=1)

        # Convert to PyTorch Linear layer
        w = torch.tensor(adjusted_weights, dtype=torch.float32)
        b = torch.tensor(adjusted_bias, dtype=torch.float32)
        logreg_layer = torch.nn.Linear(w.shape[0], w.shape[1])
        logreg_layer.weight.data = w
        logreg_layer.bias.data = b

        if return_metrics:
            return results, logreg_layer
        else:
            return logreg_layer
