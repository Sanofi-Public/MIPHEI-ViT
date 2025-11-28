"""
Test a trained model on a test dataset.

Used by run_test.py.
"""

import logging
import os
import json

import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer

import pyvips  # Avoid pyvips import error from src.dataset

from .dataset import (
    NormalizationLayer,
    get_effective_width_height,
    DataModule,
    get_width_height,
)
from .metrics import CellMetrics
from .models import ModelModule, DiscriminatorPatch
from .loss import WeightedMSELoss, CellLoss
from .generators import get_generator


def test_model(cfg: DictConfig, checkpoint_path: str, run_name: str) -> None:
    """
    Test a trained model on a test dataset using the provided configuration and checkpoint.

    This function sets up the data module, loads the model and its weights, prepares the loss
    functions, and evaluates the model on the test set.
    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all experiment parameters,
            including data paths, model settings, training options, and loss function parameters.
        checkpoint_path (str): Path to the full Lightning model checkpoint file to load for
            evaluation.
        run_name (str): Name of the current run, used for logging and experiment tracking.
    Returns:
        None
    Raises:
        FileNotFoundError: If any of the required data files or checkpoint files are not found.
        ValueError: If configuration parameters are invalid or inconsistent.
    """
    logging.getLogger('pyvips').setLevel(logging.WARNING)  # Suppress pyvips useless warnings
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: {}".format(device))

    train_dataframe = pd.read_csv(cfg.data.train_dataframe_path)
    val_dataframe = pd.read_csv(cfg.data.val_dataframe_path)
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    log.info("{} train tiles / {} val tiles / {} test tiles".format(
            len(train_dataframe), len(val_dataframe), len(test_dataframe)))
    from_slide = "image_path" not in train_dataframe.columns
    if cfg.data.slide_dataframe_path is None:
        slide_dataframe = None
    else:
        slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)

    with open(cfg.data.channel_stats_path, "r") as f:
        channel_stats = json.load(f)

    width, height = get_width_height(train_dataframe)
    width, height = get_effective_width_height(width, height, train=True)
    nc_out = len(cfg.data.targ_channel_names)
    nc_in = 3
    log.info("{} width / {} height".format(width, height))
    log.info("{} inputs channels / {} output channels".format(nc_in, nc_out))

    channel_stats_rgb = {"mean": cfg.data.normalization.mean,
                         "std": cfg.data.normalization.std}
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    channel_names = cfg.data.targ_channel_names
    targ_channel_idxs = [channel_stats[channel_name]["idx_channel"]
                         for channel_name in channel_names]
    preprocess_target_fn = NormalizationLayer(mode="if")

    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe,
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=targ_channel_idxs, from_slide=from_slide,
        input_shape=(width, height),
        batch_size=cfg.train.batch_size, pin_memory=device != "cpu",
        return_nuclei=cfg.train.use_cell_metrics, train_sampler=None,
        preprocess_input_fn=preprocess_input_fn, preprocess_target_fn=preprocess_target_fn,
        )
    data_module.setup()
    _, _, test_dataloader = data_module.get_dataloaders()

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, width, nc_in, nc_out, cfg)

    if os.name == 'nt':
        jit_compile = False
    else:
        # generator = torch.compile(generator, mode="max-autotune")
        # jit_compile = True
        jit_compile = False

    gan_train = cfg.train.gan_train
    selected_channels = [
        channel_stats[channel_name]["is_structural"] for channel_name in channel_names] \
        if cfg.train.gan_mode == "structural" else None
    discriminator = DiscriminatorPatch(
            input_nc=nc_out + nc_in, norm_layer_type=None,
            selected_channels=selected_channels) if gan_train else None
    if cfg.train.use_cell_metrics:
        cell_metrics = CellMetrics(
            target_csv_path=cfg.data.nuclei_dataframe_path,
            pred_marker_names=cfg.data.targ_channel_names,
            target_names=cfg.data.nuclei_classes,
            min_area=10)
    else:
        cell_metrics = None

    lambda_factor = cfg.train.losses.lambda_factor
    # loss_reconstruct = get_mse_loss(lambda_factor)
    # loss_reconstruct = get_focal_loss(lambda_factor, marker_weights)
    # loss_reconstruct = get_focal_loss(lambda_factor, marker_weights)
    # loss_reconstruct = L1_L2_Loss(lambda_factor=10.)
    marker_weights = torch.Tensor([channel_stats[channel_name]["std"] ** 2
                                   for channel_name in channel_names])  # Channel variances
    marker_weights = 1 / marker_weights
    marker_weights = marker_weights / marker_weights.min()
    print(marker_weights)
    loss_reconstruct = WeightedMSELoss(lambda_factor, marker_weights)

    cell_loss_params = cfg.train.losses.cell_loss
    if cell_loss_params.use_loss:
        cell_loss = CellLoss(
            cell_loss_params.mlp_path, nc_out, use_mse=cell_loss_params.use_mse,
            use_clustering=cell_loss_params.use_clustering, lambda_factor=lambda_factor)
    else:
        cell_loss = None

    # foreground_loss = CombinedBCEAndDiceLoss(1.)
    pl_model = ModelModule(
        generator=generator, discriminator=discriminator,
        lr_g=0.,
        lr_d=0.,
        cell_metrics=cell_metrics,
        cell_loss=cell_loss,
        loss_reconstruct=loss_reconstruct,
        gan_train=gan_train)

    pl_model = pl_model.to(device)
    if jit_compile:
        pl_model = torch.compile(pl_model)

    trainer = Trainer(callbacks=None, logger=None,
                      accelerator="gpu", precision=cfg.train.precision, devices=1,
                      )
    trainer.test(pl_model, test_dataloader, ckpt_path=checkpoint_path, verbose=True)
