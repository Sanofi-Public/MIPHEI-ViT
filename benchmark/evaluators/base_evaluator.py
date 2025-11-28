from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor

from src.metrics import CellMetrics
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .utils import PixelPearsonCorrCoef, get_celltype_metrics_df, save_batch


class BaseEvaluator(ABC):
    """
    Base evaluation class.
    Subclasses implement dataset/model-specific methods.
    """

    def __init__(self, checkpoint_dir, device="cuda", save_logreg=False,
                 config_dir="../configs", min_area=20, num_workers=6, batch_size=16):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.min_area = min_area
        self.save_logreg = save_logreg
        self.config_dir = Path(config_dir)

        self.num_workers = num_workers
        self.batch_size = batch_size

        self._load_config()
        self._load_marker_metadata()
        self._build_datasets()
        self._build_loaders()
        self._build_model()
        self._build_metrics()

    def _load_config(self):
        self.cfg = OmegaConf.load(self.config_dir / "data" / f"{self.dataset_name}.yaml")
        self.cfg_model = OmegaConf.load(self.checkpoint_dir / "config.yaml")

    @abstractmethod
    def _load_marker_metadata(self):
        """Load slide/val/test dataframe & dataset metadata"""
        raise NotImplementedError

    @abstractmethod
    def _build_datasets(self):
        """Instantiate datasets (val + test)"""
        raise NotImplementedError

    @abstractmethod
    def _build_model(self):
        """Instantiate model"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch):
        """Forward pass → returns predicted tensor + optional target"""
        raise NotImplementedError

    def _build_loaders(self):
        if self.val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                shuffle=False, drop_last=False, pin_memory=True
            )
        else:
            self.val_loader = None
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=False, pin_memory=True
        )

    def _build_metrics(self):
        self.pixel_metrics = {
            name: MetricCollection({
                "psnr": PeakSignalNoiseRatio(data_range=(-0.9, 0.9)),
                "ssim": StructuralSimilarityIndexMeasure(data_range=(-0.9, 0.9)),
                "pearson_r": PixelPearsonCorrCoef().to(torch.float64),
            }).to(self.device)
            for name in self.marker_names
        }

        self.cell_metrics = CellMetrics(
            target_csv_path=self.nuclei_dataframe_path,
            pred_marker_names=self.marker_names,
            target_names=self.nuclei_classes,
            train_test_split_fn=self._cell_dataframe_train_test_split,
            min_area=self.min_area
        ).to(self.device)

    def evaluate(self):
        if self.val_loader is not None:
            self._eval_split(self.val_loader, compute_pixel=False)
        self._eval_split(self.test_loader, compute_pixel=True)

        self._postprocess()

    def _eval_split(self, loader, compute_pixel=True):
        for batch in tqdm(loader):
            pred = self.forward(batch)
            nuclei = batch["nuclei"].to(self.device)
            slide_names = batch["slide_name"]

            self.cell_metrics.update(pred, nuclei, slide_names)

            if compute_pixel and self.pixel_metrics is not None:
                target = batch["target"].to(self.device)
                pred_clipped = pred.clamp(-0.9, 0.9)
                for i, name in enumerate(self.marker_names):
                    self.pixel_metrics[name].update(
                        pred_clipped[:, [i], :, :],
                        target[:, [i], :, :],
                    )

    def _postprocess(self):
        """Save pixel metrics, run classifiers, write CSVs"""

        # Pixel level (if target was available)
        if self.pixel_metrics is not None:
            pixel_rows = []
            for name in self.marker_names:
                scores = self.pixel_metrics[name].compute()
                # scores = {"psnr": val, "ssim": val}
                row = {"marker": name}
                row.update({k: float(v.cpu()) for k, v in scores.items()})
                pixel_rows.append(row)

            pd.DataFrame(pixel_rows).to_csv(
                self.checkpoint_dir / f"{self.dataset_name}_pixel_metrics.csv",
                index=False
            )

        # Split train/test
        results_torch, cell_df = self.cell_metrics.compute(return_dataframe=True)
        if self.save_logreg:
            torch.save(results_torch["state_dict"], self.checkpoint_dir / f"{self.dataset_name}_logreg.pth")
        results_logreg_df = get_celltype_metrics_df(results_torch, self.nuclei_classes)
        results_logreg_df.to_csv(self.checkpoint_dir / f"{self.dataset_name}_logreg.csv", index=False)

        # Save cell dataframe with train/test split info
        train_indexes = self._cell_dataframe_train_test_split(cell_df)[0].index
        cell_df["split"] = "test"
        cell_df.loc[train_indexes, "split"] = "train"
        cell_df["split"] = cell_df["split"].astype("category")
        cell_df["slide_name"] = cell_df["slide_name"].astype("category")
        cell_df.to_parquet(
            self.checkpoint_dir / f"{self.dataset_name}_cell_dataframe_logreg.parquet",
            index=False, compression=None)

    def _cell_dataframe_train_test_split(self, cell_df):
        raise NotImplementedError


class BaseInference(BaseEvaluator):

    def __init__(self, pred_dir, num_workers=4, *args, **kwargs):
        self.pred_dir = pred_dir
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        super().__init__(*args, **kwargs)

    def _build_metrics(self):
        pass

    def evaluate(self):
        pass

    def _eval_split(self, loader):
        pass

    def _postprocess(self):
        pass

    def inference(self):
        if self.val_loader is not None:
            self._inference_split(self.val_loader)
        self._inference_split(self.test_loader)

    def _inference_split(self, loader):
        for batch in tqdm(loader):
            pred = self.forward(batch)
            save_batch(pred, self.executor, batch["tile_name"], self.pred_dir)
