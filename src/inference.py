"""Inference of mIF from H&E patches. Predicted TIFF files will be saved in output_dir."""

import logging
import os

import json
from pathlib import Path
from omegaconf import DictConfig

import albumentations as A
import pandas as pd
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
import torch

import pyvips  # Avoid pyvips import error from src.dataset

from .dataset import (
    NormalizationLayer,
    get_effective_width_height,
    TileSlideDataset,
    get_width_height,
)
from .models import ModelModule
from .generators.hemit_models import resize_embed_hemit_statedict
from .callbacks import SavePredictionsCallback
from .generators import get_generator
from .utils import validate_load_info, get_generator_state_dict


def inference_model(cfg: DictConfig, checkpoint_dir: str, output_dir: str) -> None:
    """
    Run inference from a trained model using the provided configuration and checkpoints.

    This function loads a model checkpoint, prepares the test dataset, applies necessary
    preprocessing and augmentations, and performs inference to generate predictions,
    which are saved as multi channel TIFF files to the specified output directory.

    Args:
        cfg (oDictConfig): Configuration associated with the trained model containing data
            paths, model hyperparameters, and training/inference settings.
        checkpoint_dir (str or Path): Directory containing the model checkpoint files
            (either .safetensors or .ckpt).
        output_dir (str or Path): Directory where the inference predictions will be saved.
    Returns:
        None
    Raises:
        FileNotFoundError: If required files such as the test dataframe or checkpoint are missing.
        RuntimeError: If there is an error during model loading or inference.
    """
    logging.getLogger('pyvips').setLevel(logging.WARNING)  # Suppress pyvips useless warnings
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: {}".format(device))

    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    log.info("{} test tiles".format(
        len(test_dataframe)))
    from_slide = "image_path" not in test_dataframe.columns
    if cfg.data.slide_dataframe_path is None:
        slide_dataframe = None
    else:
        slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)

    width, height = get_width_height(test_dataframe)
    width, height = get_effective_width_height(width, height, train=True)

    spatial_augmentations = A.Compose([
        A.CenterCrop(width=width, height=height),
    ], additional_targets={"image_target": "image", "nuclei": "image"})
    nc_out = len(cfg.data.targ_channel_names)
    nc_in = 3
    log.info("{} width / {} height".format(width, height))
    log.info("{} inputs channels / {} output channels".format(nc_in, nc_out))

    channel_stats_rgb = {"mean": cfg.data.normalization.mean,
                         "std": cfg.data.normalization.std}
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    if from_slide:
        from slidevips.torch_datasets import SlideDataset
        dataset = SlideDataset(slide_dataframe=slide_dataframe, dataframe=test_dataframe)
    else:
        dataset = TileSlideDataset(
            dataframe=test_dataframe, preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=spatial_augmentations)

    num_workers = os.cpu_count() - 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train.batch_size, pin_memory=device != "cpu",
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, width, nc_in, nc_out, cfg)
    use_safetensors = (Path(checkpoint_dir) / "model.safetensors").exists()
    if use_safetensors:
        from safetensors.torch import load_file
        checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
        state_dict = load_file(checkpoint_path, device="cpu")
        strict_load = False
        print("Loading checkpoint from safetensors")
    else:
        checkpoint_path = str(Path(checkpoint_dir) / "model.weights.ckpt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = get_generator_state_dict(state_dict)
        strict_load = True
        print("Loading checkpoint from ckpt")
    if hasattr(generator, "swinT"):
        state_dict = resize_embed_hemit_statedict(state_dict, generator)

    load_info = generator.load_state_dict(state_dict, strict=strict_load)
    if use_safetensors:
        validate_load_info(load_info)

    if os.name == 'nt':
        jit_compile = False
    else:
        # generator = torch.compile(generator)
        # jit_compile = True
        jit_compile = False

    discriminator = None
    # foreground_loss = CombinedBCEAndDiceLoss(1.)
    pl_model = ModelModule(
        generator=generator, discriminator=discriminator,
        lr_g=0.,
        lr_d=0.,
        cell_metrics=None,
        cell_loss=None,
        loss_reconstruct=None,
        gan_train=False)

    callbacks = [
        SavePredictionsCallback(output_dir)
    ]

    pl_model = pl_model.to(device)
    if jit_compile:
        pl_model = torch.compile(pl_model)

    trainer = Trainer(callbacks=callbacks, inference_mode=True,
                      accelerator="gpu", precision=cfg.train.precision, devices=1,
                      )
    trainer.predict(pl_model, dataloader)
