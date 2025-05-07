import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


class CellMetrics(Metric):
    def __init__(self, slide_dataframe, marker_names, min_area=20, **kwargs):
        super().__init__(dist_sync_on_step=False, compute_on_cpu=True, **kwargs)
        excluded_markers = ["Hoechst", "Dapi"]
        filtered_names = [(i, name) for i, name in enumerate(marker_names) if name not in excluded_markers]
        self.marker_names = [name for _, name in filtered_names]
        self.marker_idxs = [idx for idx, _ in filtered_names]
        self.intensity_prop_col_names = [f"mean_intensity-{idx}" for idx in range(len(self.marker_names))]
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

    def update(self, preds, nuclei_masks, slide_names) -> None:

        preds = torch.clip(preds[:, self.marker_idxs], -0.9, 0.9).float()
        preds = (preds + 0.9) / 1.8

        nuclei_masks = torch.unsqueeze(nuclei_masks, dim=1).float()

        num_channels = len(self.marker_names)
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

            # Apply the mask to pred and target, flatten and preserve the channel dimension
            pred_flat = pred_b.permute(1, 2, 0)[nuclei_binary]
            pred_sums = torch.zeros(
                (unique_labels.shape[0], num_channels), dtype=preds.dtype,
                device=preds.device).scatter_add_(
                    0, inverse_indices.unsqueeze(1).expand(-1, num_channels), pred_flat)
            region_counts = torch.zeros(
                unique_labels.shape[0], dtype=torch.float32,
                device=nuclei_masks.device).scatter_add_(
                    0, inverse_indices, torch.ones_like(nuclei_flat, dtype=torch.float32))


            unique_labels = unique_labels.to(torch.uint32).cpu()
            region_counts = torch.unsqueeze(region_counts.to(torch.uint16).cpu(), dim=-1)
            pred_sums = (pred_sums * 255).to(torch.uint32).cpu()

            getattr(self, f"{slide_name}_cell_id").append(unique_labels)
            getattr(self, f"{slide_name}_sum").append(pred_sums)
            getattr(self, f"{slide_name}_area").append(region_counts)

    def compute(self, logreg_layer=None, return_dataframe=False):
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
                logreg_probs = torch.sigmoid(logreg_layer(torch.from_numpy(preds).to(logreg_device)))
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
            auc = torch.tensor(roc_auc_score(y_true=targets_marker, y_score=preds_marker), dtype=torch.float32)
            auc_logreg = torch.tensor(roc_auc_score(y_true=targets_marker, y_score=logreg_probs_marker), dtype=torch.float32)
            balanced_acc = torch.tensor(balanced_accuracy_score(
                y_true=targets_marker, y_pred=logreg_preds_marker), dtype=torch.float32)
            f1 = torch.tensor(f1_score(y_true=targets_marker, y_pred=logreg_preds_marker), dtype=torch.float32)

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

    def get_dataframe_cell_pred(self):
        dataframe = []
        for slide_name in self.slide_names:
            dataframe_slide = pd.DataFrame()
            cell_ids = self.metric_state[f"{slide_name}_cell_id"]
            if len(cell_ids) == 0:
                continue
            cell_ids = torch.hstack(cell_ids).numpy() # dim_zero_cat
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

    def get_dataframe_cell_target(self, slide_names=None):
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

    def get_dataframe_cell_pred_target(self):
        dataframe = self.get_dataframe_cell_pred()
        dataframe_target = self.get_dataframe_cell_target(
            slide_names=dataframe["slide_name"].unique())

        dataframe = dataframe.merge(
            dataframe_target, left_on=["slide_name", "cell_id"],
            right_on=["slide_name", "label"], how="left")

        dataframe = dataframe.drop(columns=["area"])
        dataframe = dataframe[~dataframe["label"].isna()]
        dataframe[self.marker_cols].astype(bool).fillna(False, inplace=True)
        dataframe[self.marker_cols] = dataframe[self.marker_cols].astype(bool)
        return dataframe

    def train_logistic_regression(self, train_dataframe, test_dataframe=None, return_metrics=True):
        """
        Train logistic regression models using OneVsRestClassifier for each marker.
        If test_dataframe is not provided, it will use the training data as the test set.
        Returns:
            - results: List of tuples with (marker_target, roc_auc, balanced_acc, f1)
            - w: Adjusted weights as a PyTorch tensor
            - b: Adjusted biases as a PyTorch tensor
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
        weights = np.vstack([est.coef_.flatten() if hasattr(est, "coef_") else \
                             np.zeros(len(self.marker_cols)) for est in model.estimators_]) # avoid constant model error
        bias = np.hstack([est.intercept_.flatten() if hasattr(est, "intercept_") else \
                             0. for est in model.estimators_])
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


def find_best_threshold(y_true, y_pred, low=0, high=10, tol=1e-3):
    best_thresh = None
    best_score = 0

    while (high - low) > tol:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        score1 = balanced_accuracy_score(y_true=y_true, y_pred=y_pred > mid1)
        score2 = balanced_accuracy_score(y_true=y_true, y_pred=y_pred > mid2)

        if score1 > score2:
            high = mid2
            if score1 > best_score:
                best_score = score1
                best_thresh = mid1
        else:
            low = mid1
            if score2 > best_score:
                best_score = score2
                best_thresh = mid2

    return best_thresh, best_score
