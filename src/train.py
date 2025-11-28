"""
Train a model for H&E to mIF image translation.

Used by run.py.
"""

import logging
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import pyvips  # Avoid pyvips import error from src.dataset

from .dataset import (
    NormalizationLayer,
    DataModule,
    BalancedPositiveSampler,
    get_width_height,
    get_effective_width_height,
)
from .metrics import CellMetrics
from .models import ModelModule, DiscriminatorPatch
from .utils import wandb_log_artifact, update_wandb_note
from .callbacks import (
    WandbVisCallback,
    SlideAugmentationCallback,
    DebugImageLogger,
    TileAugmentationCallback,
)
from .loss import WeightedMSELoss, CellLoss
from .generators import get_generator


def train_miphei(cfg: DictConfig, logdir: str) -> None:
    """
    Train a MIPHEI model using the provided configuration and logging directory.

    This function sets up data loaders, model components (generator, discriminator), loss functions,
    and training callbacks for a MIPHEI-based image-to-image translation task. It supports various
    loss types, data sampling strategies, and logging with Weights & Biases (wandb). The function
    also handles model checkpointing and optional cell metrics computation.
    Args:
        cfg (omegaconf.DictConfig): Hydra configuration object from configs folder containing all
            training, data, and model parameters.
        logdir (str or Path): Directory path for saving logs, checkpoints, and configuration files.
    Raises:
        FileNotFoundError: If any of the required data files specified in the configuration are
            missing.
        ValueError: If configuration parameters are invalid or inconsistent.
    Returns:
        None
    """
    logging.getLogger('pyvips').setLevel(logging.WARNING)  # Suppress pyvips useless warnings
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: {}".format(device))

    logdir = Path(logdir)
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

    sampler_cfg = cfg.train.data_sampler
    if sampler_cfg.use_sampler:
        train_sampler = BalancedPositiveSampler(
            train_dataframe, channel_names, sampler_cfg.tresh,
            other_percent=sampler_cfg.other_percent)
    else:
        train_sampler = None

    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe,
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=targ_channel_idxs, from_slide=from_slide,
        input_shape=(width, height),
        batch_size=cfg.train.batch_size, pin_memory=device != "cpu",
        return_nuclei=cfg.train.use_cell_metrics, train_sampler=train_sampler,
        preprocess_input_fn=preprocess_input_fn, preprocess_target_fn=preprocess_target_fn,
        )
    data_module.setup()
    train_dataloader, val_dataloader, test_dataloader = data_module.get_dataloaders()

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, width, nc_in, nc_out, cfg)

    if cfg.model.checkpoint_path:
        generator.load_state_dict(torch.load(cfg.model.checkpoint_path))
        log.info("checkpoint lodaded from {}".format(cfg.model.checkpoint_path))

    if os.name == 'nt':
        jit_compile = False
    else:
        # generator = torch.compile(generator, mode="max-autotune")

        # only on encoder, because it can generate out of memory on GPU
        # allows to accelerate training
        generator.encoder = torch.compile(generator.encoder, mode="max-autotune")
        jit_compile = False

    ckpt_weights = str(logdir / "model.weights")

    log.info("PatchGAN training")

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
    gan_train = cfg.train.gan_train
    selected_channels = [
        channel_stats[channel_name]["is_structural"] for channel_name in channel_names] \
        if cfg.train.gan_mode == "structural" else None
    discriminator = DiscriminatorPatch(
            input_nc=nc_out + nc_in, norm_layer_type=None,
            selected_channels=selected_channels) if gan_train else None

    pl_model = ModelModule(
        generator=generator, discriminator=discriminator,
        lr_g=cfg.train.learning_rate_g * np.sqrt(cfg.train.batch_size),
        lr_d=cfg.train.learning_rate_d * np.sqrt(cfg.train.batch_size),
        cell_metrics=cell_metrics,
        cell_loss=cell_loss,
        loss_reconstruct=loss_reconstruct,
        gan_train=gan_train)

    logger_name = logdir.name
    wandb_note = cfg.train.wandb_note
    wandb_note = update_wandb_note(wandb_note)
    logger = WandbLogger(project=cfg.train.wandb_project, name=logger_name, notes=wandb_note,
                         log_model=False, save_dir=str(logdir), force=True, reinit=True)
    cfg_path = str(logdir / "config.yaml")
    OmegaConf.save(cfg, cfg_path)
    wandb_log_artifact(logger, "cfg", "config", cfg_path)
    wandb_log_artifact(logger, "stats_image", "stats", cfg.data.channel_stats_path)
    wandb_log_artifact(logger, "gitlog", "gitlog", str(logdir / "github_log.txt"))

    config_callback = cfg.train.callbacks
    ckpt_dirpath = str(Path(ckpt_weights).parent)
    ckpt_filename = str(Path(ckpt_weights).name)
    callbacks = [
        DebugImageLogger("logs_img", batch_frequency=1000, max_images=4, clamp=True),
        ModelCheckpoint(
            dirpath=ckpt_dirpath, filename=ckpt_filename,
            monitor=config_callback.modelcheckpoint.monitor,
            save_top_k=1,
            mode=config_callback.modelcheckpoint.mode, save_last=False,
            save_weights_only=True, verbose=1),
        WandbVisCallback(preprocess_input_fn.unormalize, num_samples=4),
        # SwitchGenDiscTrain()
    ]
    if cfg.data.augmentation_dir is not None:
        if from_slide:
            callbacks.append(SlideAugmentationCallback(cfg.data.augmentation_dir, prob=0.25))
        else:
            callbacks.append(TileAugmentationCallback(cfg.data.augmentation_dir, prob=0.25))

    pl_model = pl_model.to(device)
    if jit_compile:
        pl_model = torch.compile(pl_model)

    trainer = Trainer(max_epochs=cfg.train.epochs, callbacks=callbacks, logger=logger,
                      accelerator="gpu", precision=cfg.train.precision, devices=1,
                      )  # limit_train_batches=100, limit_val_batches=100, limit_test_batches=100)
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    trainer.test(pl_model, test_dataloader, ckpt_path=ckpt_weights + ".ckpt", verbose=True)
    wandb.finish()
