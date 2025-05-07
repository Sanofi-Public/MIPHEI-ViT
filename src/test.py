"""
Test scripts for image to image segmentation.
"""
import logging
logging.getLogger('pyvips').setLevel(logging.WARNING)

import os
os.environ['VIPS_CONCURRENCY'] = '1'
import pyvips

from omegaconf import OmegaConf
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import torch

from .dataset import NormalizationLayer, get_effective_width_height, DataModule,\
    get_width_height
from .metrics import CellMetrics
from .models import ModelModule, DiscriminatorPatch
from .utils import get_foreground_weight
from .callbacks import DebugImageLogger
from .loss import WeightedMSELoss, get_focal_loss, CellLoss
from .generators import get_generator


def test_model(cfg, checkpoint_path, run_name):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    pyvips.cache_set_max(0)
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

    preprocess_input_fn = NormalizationLayer(channel_stats["RGB"], mode="he")

    channel_names = cfg.data.targ_channel_names
    targ_channel_idxs = [channel_stats[channel_name]["idx_channel"] \
                         for channel_name in channel_names]
    stats_list_if = [channel_stats[channel_name] for channel_name in channel_names].copy()
    preprocess_target_fn = NormalizationLayer(stats_list_if, mode="if")

    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe, 
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=targ_channel_idxs, from_slide=from_slide,
        input_shape=(width, height),
        batch_size=cfg.train.batch_size, pin_memory=device!="cpu",
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
        #generator = torch.compile(generator)
        #jit_compile = True
        jit_compile = False


    gan_train = cfg.train.gan_train
    selected_channels = [
        channel_stats[channel_name]["is_structural"] for channel_name in channel_names] \
            if cfg.train.gan_mode == "stuctural" else None
    discriminator = DiscriminatorPatch(
            input_nc=nc_out + nc_in, norm_layer_type=None,
            selected_channels=selected_channels) if gan_train else None
    if cfg.train.use_cell_metrics:
        cell_metrics = CellMetrics(slide_dataframe, channel_names)
    else:
        cell_metrics = None


    lambda_factor = cfg.train.losses.lambda_factor
    if cfg.train.losses.use_weighted_mae:
        foreground_weight = get_foreground_weight(
            channel_names, train_dataframe)
        foreground_weight = np.float32(foreground_weight)
        foreground_weight = torch.tensor(foreground_weight).reshape((1, -1, 1, 1)).to(device)
        foreground_thresh = preprocess_target_fn(0)

        print("foreground_weight", foreground_weight.cpu().numpy().flatten().tolist(),
            "foreground_thresh", foreground_thresh.flatten().tolist())
        #loss_reconstruct = get_weighted_mae_loss(lambda_factor, foreground_weight, foreground_thresh)
        loss_reconstruct = get_focal_loss(lambda_factor, foreground_weight)
        #loss_reconstruct = get_shrinkage_loss(lambda_factor, foreground_weight)
    else:
        #loss_reconstruct = get_mse_loss(lambda_factor)
        marker_weights = torch.Tensor([channel_stats[channel_name]["std"] \
                         for channel_name in channel_names])
        marker_weights = 1 / marker_weights
        marker_weights = marker_weights / marker_weights.min()
        print(marker_weights)
        loss_reconstruct = WeightedMSELoss(lambda_factor, marker_weights)
        #loss_reconstruct = L1_L2_Loss(lambda_factor=10.)

    cell_loss_params = cfg.train.losses.cell_loss
    if cell_loss_params.use_loss:
        cell_loss = CellLoss(
            cell_loss_params.mlp_path, nc_out, use_mse=cell_loss_params.use_mse,
            use_clustering=cell_loss_params.use_clustering, lambda_factor=lambda_factor)
    else:
        cell_loss = None

    #foreground_loss = CombinedBCEAndDiceLoss(1.)
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
