import pyvips  # avoid errors

import argparse
import torch
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from benchmark.models import get_rosie
from benchmark.models.rosie import infer_sliding_window
from benchmark.evaluators.base_evaluator import BaseInference
from benchmark.evaluators.dataset_evaluator import DATASET_EVALUATOR_BASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Path to directory with saved predictions")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    return parser.parse_args()


class RosieInference(BaseInference):

    def __init__(self, patch_size=128, stide=8, *args, **kwargs):
        self.patch_size = patch_size
        assert self.patch_size % 2 == 0, "Patch size must be even."
        self.stride = stide
        pad = patch_size // 2
        self.padding = (pad, pad, pad, pad)
        super().__init__(*args, **kwargs)

    def _load_config(self):
        super()._load_config()

        self.input_mean = self.cfg_model.data.normalization.mean
        self.input_std = self.cfg_model.data.normalization.std

    def _build_model(self):
        """Instantiate model"""
        torch.cuda.empty_cache()
        self.model = get_rosie(self.checkpoint_dir, self.cfg_model, self.device)
        self.model.to(self.device).eval()

    def forward(self, batch):
        """No model is loaded: predictions already exist on disk."""
        input = batch["image"].to(self.device)
        input = torch.nn.functional.pad(
            input, self.padding, mode="constant", value=0)
        out = infer_sliding_window(
            input, self.model,
            P=self.patch_size, S=self.stride).numpy()
        out = np.moveaxis(out, 1, -1)
        return out


def get_inference_class(dataset_name: str):
    dataset_base = DATASET_EVALUATOR_BASES[dataset_name]
    class_name = f"{dataset_name.capitalize()}RosieInference"
    return type(class_name, (RosieInference, dataset_base), {})


if __name__ == "__main__":

    args = parse_args()

    Path(args.pred_dir).mkdir(exist_ok=True)
    inference_rosie_class = get_inference_class(args.dataset)
    inference_rosie = inference_rosie_class(
        checkpoint_dir=args.checkpoint_dir,
        pred_dir=args.pred_dir,
        device=args.device,
        config_dir="configs",
        num_workers=args.num_workers
    )

    inference_rosie.inference()
