import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import albumentations as A

from .base_evaluator import BaseEvaluator
from ..data.dataset import LizardDataset, TileDataset
from .utils import build_pred_target_mapping
from src.dataset import NormalizationLayer, TileImg2ImgSlideDataset, Img2ImgNucleiSlideDataset
from src.dataset import get_width_height, get_effective_width_height


class OrionBaseEvaluator(BaseEvaluator):
    dataset_name = "orion"

    def _load_marker_metadata(self):

        target_channel_names = pd.read_csv(self.cfg.data.marker_metadata_path)["Marker Name"].tolist()
        marker_metadata_df = pd.read_csv(self.checkpoint_dir / "marker_metadata.csv")
        common_markers, pred_indices, targ_indices = build_pred_target_mapping(
            target_channel_names, self.dataset_name, marker_metadata_df)
        self.marker_names = common_markers
        self.target_marker_names = [target_channel_names[idx] for idx in targ_indices]
        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path
        self.targ_channel_idxs = targ_indices
        self.pred_indices = pred_indices
        print(self.marker_names)

    def _build_datasets(self):
        self.slide_dataframe = pd.read_csv(self.cfg.data.slide_dataframe_path)
        self.val_dataframe = pd.read_csv(self.cfg.data.val_dataframe_path)
        self.test_dataframe = pd.read_csv(self.cfg.data.test_dataframe_path)

        width, height = get_width_height(self.test_dataframe)
        width_crop, height_crop = get_effective_width_height(width, height, train=True)

        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width_crop, height=height_crop),
        ], additional_targets={"image_target": "image", "nuclei": "image"})

        channel_stats_rgb = {
            "mean": self.input_mean,
            "std": self.input_std}
        preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")
        preprocess_target_fn = NormalizationLayer(mode="if")

        self.val_dataset = TileImg2ImgSlideDataset(
            dataframe=self.val_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            targ_channel_idxs=self.targ_channel_idxs,
            spatial_augmentations=spatial_augmentations, return_nuclei=True)
        self.test_dataset = TileImg2ImgSlideDataset(
            dataframe=self.test_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            targ_channel_idxs=self.targ_channel_idxs,
            spatial_augmentations=spatial_augmentations, return_nuclei=True)

    def _cell_dataframe_train_test_split(self, cell_df):
        val_names = set(self.val_dataframe["in_slide_name"].unique())
        test_names = set(self.test_dataframe["in_slide_name"].unique())

        train_cell_df = cell_df[cell_df["slide_name"].isin(val_names)]
        test_cell_df = cell_df[cell_df["slide_name"].isin(test_names)]

        return train_cell_df, test_cell_df


class HEMITBaseEvaluator(BaseEvaluator):
    dataset_name = "hemit"

    def _load_marker_metadata(self):

        target_channel_names = pd.read_csv(self.cfg.data.marker_metadata_path)["Marker Name"].tolist()
        marker_metadata_df = pd.read_csv(self.checkpoint_dir / "marker_metadata.csv")
        common_markers, pred_indices, targ_indices = build_pred_target_mapping(
            target_channel_names, self.dataset_name, marker_metadata_df)
        self.marker_names = common_markers
        self.target_marker_names = [target_channel_names[idx] for idx in targ_indices]
        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path
        self.targ_channel_idxs = targ_indices
        self.pred_indices = pred_indices
        print(self.marker_names)

    def _build_datasets(self):
        self.slide_dataframe = None
        self.val_dataframe = None
        self.test_dataframe = pd.concat((
            pd.read_csv(self.cfg.data.train_dataframe_path),
            pd.read_csv(self.cfg.data.val_dataframe_path),
            pd.read_csv(self.cfg.data.test_dataframe_path)))

        width, height = 1024, 1024
        # Hemit is at 40x -> downsample by 2 -> 20x
        width_crop, height_crop = width // 2, height // 2
        spatial_augmentations = A.Compose([
            A.Resize(width=width_crop, height=height_crop), # linear interp for image, nearest for mask
            A.CenterCrop(width=width_crop, height=height_crop),
        ], additional_targets={"image_target": "image", "nuclei": "mask"})

        channel_stats_rgb = {
            "mean": self.input_mean,
            "std": self.input_std}
        preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")
        preprocess_target_fn = NormalizationLayer(mode="if")

        self.val_dataset = None
        self.test_dataset = TileImg2ImgSlideDataset(
            dataframe=self.test_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            targ_channel_idxs=self.targ_channel_idxs,
            spatial_augmentations=spatial_augmentations, return_nuclei=True)

    def _cell_dataframe_train_test_split(self, cell_df):
        target_cols = self.nuclei_classes
        combos = cell_df[target_cols].apply(tuple, axis=1)
        stratify, _ = pd.factorize(combos)
        train_cell_df, test_cell_df = train_test_split(
            cell_df, stratify=stratify, test_size=0.8, random_state=42)
        return train_cell_df, test_cell_df


class PathocellBaseEvaluator(BaseEvaluator):
    dataset_name = "pathocell"

    def _load_marker_metadata(self):

        target_channel_names = pd.read_csv(self.cfg.data.marker_metadata_path)["Marker Name"].tolist()
        marker_metadata_df = pd.read_csv(self.checkpoint_dir / "marker_metadata.csv")
        common_markers, pred_indices, targ_indices = build_pred_target_mapping(
            target_channel_names, self.dataset_name, marker_metadata_df)
        self.marker_names = common_markers
        self.target_marker_names = [target_channel_names[idx] for idx in targ_indices]
        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path
        self.targ_channel_idxs = targ_indices
        self.pred_indices = pred_indices
        print(self.marker_names)

    def _build_datasets(self):
        self.slide_dataframe = pd.read_csv(self.cfg.data.slide_dataframe_path)
        self.val_dataframe = None
        self.test_dataframe = pd.read_csv(self.cfg.data.test_dataframe_path)

        width, height = get_width_height(self.test_dataframe)
        width_crop, height_crop = get_effective_width_height(width, height, train=True)

        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width_crop, height=height_crop),
        ], additional_targets={"image_target": "image", "nuclei": "image"})

        channel_stats_rgb = {
            "mean": self.input_mean,
            "std": self.input_std}
        preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")
        preprocess_target_fn = NormalizationLayer(mode="if")

        self.val_dataset = None
        self.test_dataset = Img2ImgNucleiSlideDataset(
            slide_dataframe=self.slide_dataframe,
            dataframe=self.test_dataframe, preprocess_input_fn=preprocess_input_fn,
            preprocess_target_fn=preprocess_target_fn,
            targ_channel_idxs=self.targ_channel_idxs,
            mode_targ="IF",
            spatial_augmentations=spatial_augmentations, return_nuclei=True)

    def _cell_dataframe_train_test_split(self, cell_df):
        # one true label per row
        stratify = cell_df[self.nuclei_classes].values.argmax(axis=1)
        train_cell_df, test_cell_df = train_test_split(
            cell_df, stratify=stratify, test_size=0.8, random_state=42)
        return train_cell_df, test_cell_df


class LizardBaseEvaluator(BaseEvaluator):
    dataset_name = "lizard"

    def _build_metrics(self):
        super()._build_metrics()
        self.pixel_metrics = None # no pixel metrics for lizard

    def _load_marker_metadata(self):
        self.marker_names = pd.read_csv(
            self.checkpoint_dir / "marker_metadata.csv")["predicted_marker"].tolist()
        self.pred_indices = None

        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path

    def _build_datasets(self):
        self.slide_dataframe = pd.read_csv(self.cfg.data.slide_dataframe_path)
        self.val_dataframe = None
        self.test_dataframe = pd.read_csv(self.cfg.data.test_dataframe_path)

        width, height = get_width_height(self.test_dataframe)
        width_crop, height_crop = get_effective_width_height(width, height, train=True)

        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width_crop, height=height_crop),
        ], additional_targets={"image_target": "image", "nuclei": "image"})

        channel_stats_rgb = {
            "mean": self.input_mean,
            "std": self.input_std}
        preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

        self.val_dataset = None
        self.test_dataset = LizardDataset(
            slide_dataframe=self.slide_dataframe,
            dataframe=self.test_dataframe, mpp=0.5,
            preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=spatial_augmentations,
            return_nuclei=True)

    def _cell_dataframe_train_test_split(self, cell_df):
        stratify = cell_df[self.nuclei_classes].values.argmax(axis=1)
        train_cell_df, test_cell_df = train_test_split(
            cell_df, stratify=stratify, test_size=0.8, random_state=42)
        return train_cell_df, test_cell_df


class PannukeBaseEvaluator(BaseEvaluator):
    dataset_name = "pannuke"

    def _build_metrics(self):
        super()._build_metrics()
        self.pixel_metrics = None # no pixel metrics for pannuke

    def _load_marker_metadata(self):
        self.marker_names = pd.read_csv(
            self.checkpoint_dir / "marker_metadata.csv")["predicted_marker"].tolist()
        self.pred_indices = None

        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path

    def _build_datasets(self):
        self.slide_dataframe = None
        self.val_dataframe = None
        self.test_dataframe = pd.read_csv(self.cfg.data.test_dataframe_path)

        width, height = get_width_height(self.test_dataframe)
        width_crop, height_crop = get_effective_width_height(width, height, train=True)

        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width_crop, height=height_crop),
        ], additional_targets={"image_target": "image", "nuclei": "image"})

        channel_stats_rgb = {
            "mean": self.input_mean,
            "std": self.input_std}
        preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

        self.val_dataset = None
        self.test_dataset = TileDataset(
            dataframe=self.test_dataframe,
            preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=spatial_augmentations,
            return_nuclei=True)

    def _cell_dataframe_train_test_split(self, cell_df):
        stratify = cell_df[self.nuclei_classes].values.argmax(axis=1)
        train_cell_df, test_cell_df = train_test_split(
            cell_df, stratify=stratify, test_size=0.8, random_state=42)
        return train_cell_df, test_cell_df


DATASET_EVALUATOR_BASES = {
    "orion":     OrionBaseEvaluator,
    "pathocell": PathocellBaseEvaluator,
    "hemit":     HEMITBaseEvaluator,
    "lizard":    LizardBaseEvaluator,
    "pannuke":   PannukeBaseEvaluator,
}
