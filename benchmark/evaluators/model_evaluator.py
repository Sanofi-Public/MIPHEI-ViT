import albumentations as A
import torch
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from slidevips import SlideVips

from .base_evaluator import BaseEvaluator
from ..models.rosie import retrieve_image_scale
from ..data.dataset import SlideVipsPNG, SlideOutputRosieDataset, TileOutputRosieDataset
from .utils import extract_cell_features
from benchmark.models import (
    get_miphei,
    get_hemit,
    get_pix2pix,
    get_diffusion_ft,
)
from src.dataset import NormalizationLayer, get_width_height, get_effective_width_height
from src.metrics import train_logistic_regression


class RosieEvaluator(BaseEvaluator):
    def __init__(self, pred_dir, *args, **kwargs):
        self.pred_dir = pred_dir
        self.input_mean = None
        self.input_std = None
        super().__init__(*args, **kwargs)

    def _rosie_adapt_dataframe(self, dataframe, is_wsi_reader):
        if is_wsi_reader:
            tile_names_row = dataframe.apply(
                lambda row: f"{row['in_slide_name']}_{row['x']}_{row['y']}"\
                    f"_{row['level']}_{row['tile_size_x']}_{row['tile_size_y']}",
                axis=1)
            dataframe["pred_path"] = tile_names_row.apply(lambda x: str(Path(self.pred_dir) / f"{x}.tiff"))
        else:
            dataframe["pred_path"] = dataframe["image_path"].apply(
                    lambda x: str(Path(self.pred_dir) / (Path(x).stem + ".tiff")))
        return dataframe

    def _build_datasets(self):
        super()._build_datasets()

        self.is_wsi_reader = any(
            c in self.test_dataframe.columns for c in [
                "x", "y", "level", "tile_size_x", "tile_size_y"])

        if self.val_dataframe is not None:
            self.val_dataframe = self._rosie_adapt_dataframe(
                self.val_dataframe, self.is_wsi_reader)
        self.test_dataframe = self._rosie_adapt_dataframe(
            self.test_dataframe, self.is_wsi_reader)

        width, height = get_width_height(self.test_dataframe)
        width_crop, height_crop = get_effective_width_height(width, height, train=True)
        width_pred_crop, height_pred_crop = width // 8, height // 8

        spatial_augmentations = A.Compose(
                [A.CenterCrop(width=width_crop, height=height_crop)],
                additional_targets={"image_target": "image", "nuclei": "image"})
        pred_augmentations = A.Compose([
            A.Lambda(partial(retrieve_image_scale, shape_crop=(width_pred_crop, height_pred_crop))),
            A.PadIfNeeded(min_height=width, min_width=height, border_mode=0),
        ])

        return_target = "target" in self.test_dataset[0].keys()
        preprocess_pred_fn = NormalizationLayer(mode="if")

        kwargs_dataset = {
            "preprocess_pred_fn": preprocess_pred_fn,
            "preprocess_target_fn": preprocess_pred_fn if return_target else None,
            "pred_channel_idxs": self.pred_indices,
            "targ_channel_idxs": self.targ_channel_idxs if return_target else None,
            "pred_augmentations": pred_augmentations,
            "spatial_augmentations": spatial_augmentations,
            "return_target": return_target,
            "return_nuclei": True
        }

        if self.is_wsi_reader:
            wsi_reader_cls = SlideVipsPNG if self.dataset_name == "lizard" else SlideVips
            wsi_reader_kwargs = {"mode": "IF"} if self.dataset_name != "lizard" else {"mpp": 0.5}
            if self.val_dataset is not None:
                self.val_dataset = SlideOutputRosieDataset(
                    slide_dataframe=self.slide_dataframe,
                    wsi_reader_kwargs=wsi_reader_kwargs,
                    wsi_reader_cls=wsi_reader_cls,
                    dataframe=self.val_dataframe,
                    **kwargs_dataset
                )

            self.test_dataset = SlideOutputRosieDataset(
                slide_dataframe=self.slide_dataframe,
                wsi_reader_kwargs=wsi_reader_kwargs,
                wsi_reader_cls=wsi_reader_cls,
                dataframe=self.test_dataframe,
                **kwargs_dataset
            )
        else:
            if self.val_dataset is not None:
                self.val_dataset = TileOutputRosieDataset(
                    dataframe=self.val_dataframe,
                    **kwargs_dataset
                )

            self.test_dataset = TileOutputRosieDataset(
                dataframe=self.test_dataframe,
                **kwargs_dataset
            )

    def _build_model(self):
        """Instantiate model"""
        pass

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        pred = batch["pred"].to(self.device)
        return pred


class MIPHEIEvaluator(BaseEvaluator):

    def _load_config(self):
        super()._load_config()
        self.cfg_model = OmegaConf.load(self.checkpoint_dir / "config.yaml")

        self.input_mean = self.cfg_model.data.normalization.mean
        self.input_std = self.cfg_model.data.normalization.std

    def _build_model(self):
        """Instantiate model"""
        torch.cuda.empty_cache()
        _, width, height = self.test_dataset[0]["image"].shape
        self.model = get_miphei(self.checkpoint_dir, self.cfg_model,
                                self.device, H=width, W=height)
        self.model.to(self.device).eval().half()

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        x = batch["image"].to(self.device)
        with torch.inference_mode():
            out = self.model(x.half()).float()
        if self.pred_indices is not None:
            out = out[:, self.pred_indices]
        return out


class DiffusionEvaluator(BaseEvaluator):

    def _load_config(self):
        super()._load_config()
        self.cfg_model = OmegaConf.load(self.checkpoint_dir / "config.yaml")

        self.input_mean = self.cfg_model.data.normalization.mean
        self.input_std = self.cfg_model.data.normalization.std

    def _build_model(self):
        """Instantiate model"""
        torch.cuda.empty_cache()
        self.model = get_diffusion_ft(self.checkpoint_dir, self.device)

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        x = batch["image"].to(self.device)
        with torch.inference_mode():
            out = self.model(x, return_torch=True)
            out = 1.8 * out - 0.9
        if self.pred_indices is not None:
            out = out[:, self.pred_indices]
        return out


class Pix2pixEvaluator(BaseEvaluator):

    def _load_config(self):
        super()._load_config()
        self.cfg_model = OmegaConf.load(self.checkpoint_dir / "config.yaml")

        self.input_mean = self.cfg_model.data.normalization.mean
        self.input_std = self.cfg_model.data.normalization.std

    def _build_model(self):
        """Instantiate model"""
        torch.cuda.empty_cache()
        self.model = get_pix2pix(self.checkpoint_dir, self.cfg_model, self.device)
        self.model.to(self.device).eval()

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        x = batch["image"].to(self.device)
        with torch.inference_mode():
            out = self.model(x)
            out = 0.9 * out.clamp(-1, 1)
        if self.pred_indices is not None:
            out = out[:, self.pred_indices]
        return out


class HEMITEvaluator(Pix2pixEvaluator):

    def _build_model(self):
        """Instantiate model"""
        torch.cuda.empty_cache()
        _, w_in, h_in = self.test_dataset[0]["image"].shape
        self.model = get_hemit(self.checkpoint_dir, self.cfg_model,
                               self.device, img_size=(w_in, h_in))
        self.model.eval().to(self.device)


class UpperBoundEvaluator(BaseEvaluator):

    def _load_config(self):
        super()._load_config()
        self.cfg_model = OmegaConf.load(self.checkpoint_dir / "config.yaml")

        self.input_mean = self.cfg_model.data.normalization.mean
        self.input_std = self.cfg_model.data.normalization.std

    def _build_metrics(self):
        super()._build_metrics()
        self.pixel_metrics = None

    def _build_model(self):
        """Instantiate model"""
        pass

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        y = batch["target"].to(self.device)
        return y


class MorphoEvaluator(BaseEvaluator):

    def _load_config(self):
        self.cfg = OmegaConf.load(self.config_dir / "data" / f"{self.dataset_name}.yaml")
        self.input_mean = [127.5, 127.5, 127.5]
        self.input_std = [127.5, 127.5, 127.5]

    def _load_marker_metadata(self):

        self.marker_names = [
            'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'mean_intensity', 'std_intensity', 'max_intensity', 'min_intensity'
        ]
        self.nuclei_classes = self.cfg.data.nuclei_classes
        self.nuclei_dataframe_path = self.cfg.data.nuclei_dataframe_path
        self.targ_channel_idxs = [0] # only DAPI as target is not used
        print(self.marker_names)

    def _build_metrics(self):
        super()._build_metrics()
        self.cell_morphology_df = []
        self.pixel_metrics = None
        self.cell_metrics = None

    def _eval_split(self, loader, compute_pixel=True):
        for batch in tqdm(loader):
            cell_df_batch = self.forward(batch)
            self.cell_morphology_df.append(cell_df_batch)

    def evaluate(self):
        if self.val_loader is not None:
            self._eval_split(self.val_loader, compute_pixel=False)
        self._eval_split(self.test_loader, compute_pixel=True)
        cell_morphology_df = pd.concat(
            self.cell_morphology_df, ignore_index=True)
        # merge potential duplicate cell
        cell_morphology_df = (cell_morphology_df.groupby(["slide_name", "label"])[self.marker_names]
                              .mean().reset_index())
        cell_target_df = pd.read_parquet(self.nuclei_dataframe_path)
        # merge prediction and targets
        cell_morphology_df = cell_morphology_df.merge(
            cell_target_df, on=["label", "slide_name"])
        
        # post process merged dataframe
        # discard if not in ground truth
        cell_morphology_df = cell_morphology_df[~cell_morphology_df["label"].isna()]
        cell_morphology_df[self.nuclei_classes].astype(bool).fillna(False, inplace=True)
        train_cell_morphology_df, test_cell_morphology_df = self._cell_dataframe_train_test_split(
            cell_morphology_df)
        results, logreg = train_logistic_regression(
                train_dataframe=train_cell_morphology_df,
                test_dataframe=test_cell_morphology_df,
                pred_cols=self.marker_names,
                target_cols=self.nuclei_classes,
                return_metrics=True)

        if self.save_logreg:
            torch.save(logreg.state_dict(), self.checkpoint_dir / f"{self.dataset_name}_logreg.pth")
        #cell_morphology_df["slide_name"] = cell_morphology_df["slide_name"].apply(lambda x: Path(x).stem)
        results_logreg_df = pd.DataFrame(
            results, columns=["Marker", "ROC AUC", "AUPRC", "Balanced Accuracy", "F1 Score"])
        results_logreg_df.to_csv(self.checkpoint_dir / f"{self.dataset_name}_morphology_logreg.csv", index=False)
        cell_morphology_df.to_parquet(
            self.checkpoint_dir / f"{self.dataset_name}_cell_morphology.parquet", index=False)

        target_cols = self.nuclei_classes
        cell_df_target = cell_morphology_df[target_cols]
        probs_dict = cell_df_target[target_cols].mean().to_dict()
        results_random = {}
        for class_name, prob in probs_dict.items():
            y_true = cell_df_target[class_name].to_numpy()
            f1s = []
            for _ in range(100):
                random_pred = (torch.rand(len(cell_df_target)) < prob).numpy()
                f1 = f1_score(y_true=y_true, y_pred=random_pred)
                f1s.append(f1)
            results_random[class_name] = {"F1 Score": sum(f1s) / len(f1s)}
        results_random_df = pd.DataFrame.from_dict(
            results_random, orient="index").reset_index().rename(columns={"index": "Marker"})
        results_random_df.to_csv(self.checkpoint_dir / f"{self.dataset_name}_random_f1.csv", index=False)

    def _build_model(self):
        """Instantiate model"""
        pass

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        x_np = batch["image"].permute(0, 2, 3, 1).numpy()
        nuclei_masks_np = batch["nuclei"].numpy()
        tile_names = batch["tile_name"]
        slide_names = batch["slide_name"]

        cell_df_batch = []
        for idx_batch in range(x_np.shape[0]):
            image_he = x_np[idx_batch]
            inst_mask = nuclei_masks_np[idx_batch]
            cell_df_i = extract_cell_features(image_he, inst_mask)
            cell_df_i["tile_name"] = tile_names[idx_batch]
            cell_df_i["slide_name"] = slide_names[idx_batch]
            cell_df_batch.append(cell_df_i)
        cell_df_batch = pd.concat(cell_df_batch, ignore_index=True)
        return cell_df_batch


MODEL_EVALUATOR_BASES = {
    "miphei": MIPHEIEvaluator,
    "rosie":  RosieEvaluator,
    "diffusion": DiffusionEvaluator,
    "pix2pix": Pix2pixEvaluator,
    "upperbound": UpperBoundEvaluator,
    "hemit": HEMITEvaluator,
    "morpho": MorphoEvaluator,
}
