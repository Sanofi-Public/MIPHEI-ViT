"""Factory functions for generator models."""

import torch
from omegaconf import DictConfig

from .smp_unet import UnetMultiHeads, UnetMultiHeadsFG
from .mipheivit import get_mipheivit
from .unetr import UnetR
from .hemit_models import get_generator_hemit


def get_generator(model_name: str, img_size: int, nc_in: int, nc_out: int, cfg: DictConfig
                  ) -> torch.nn.Module:
    """
    Create and return a generator model instance based on the specified model name and config.

    This function supports multiple generator architectures, including variants of UNet,
    MIPHEI-ViT, and HEMIT.
    Args:
        model_name (str): Name of the generator model to instantiate. Determines the architecture
            and options.
        img_size (int or tuple): Input image size for the generator.
        nc_in (int): Number of input channels.
        nc_out (int): Number of output channels.
        cfg (omegaconf.DictConfig): Configuration object containing model and training parameters.
    Returns:
        generator (nn.Module): Instantiated generator model.
    Raises:
        NotImplementedError: If the specified model_name is not supported.
    """
    if model_name.startswith("smp_unet"):
        if cfg.train.foreground_head:
            unet_class = UnetMultiHeadsFG
        else:
            unet_class = UnetMultiHeads
        generator = unet_class(
            encoder_name=cfg.model.encoder.encoder_name,
            encoder_weights=cfg.model.encoder.encoder_weights,
            decoder_use_batchnorm=True,
            dropout=cfg.model.dropout,
            in_channels=nc_in,
            classes=nc_out,
            activation=torch.nn.Tanh)

    elif model_name.startswith("unetr"):
        if cfg.train.foreground_head:
            raise NotImplementedError
        use_lora = True if "lora" in model_name else False
        generator = UnetR(
            img_size=img_size,
            encoder_name=cfg.model.encoder.encoder_name,
            encoder_weights=cfg.model.encoder.encoder_weights,
            decoder_out_channels=32,
            head_use_attention=True,
            use_lora=use_lora,
            classes=nc_out,
            drop_rate=cfg.model.dropout,
            activation=torch.nn.Tanh()
            )
        if "frozen" in model_name:
            generator.freeze_encoder()

    elif model_name.startswith("mipheivit"):
        ckpt_path = cfg.model.encoder.encoder_weights
        generator = get_mipheivit(
            cfg.model.encoder.encoder_name, img_size, nc_out, use_lora=True, ckpt_path=ckpt_path)

    elif model_name.startswith("hemit"):
        generator = get_generator_hemit(
            input_nc=nc_in, output_nc=nc_out, image_size=img_size, ngf=64,
            netG="SwinTResnet", norm='batch', use_dropout=False)

    else:
        raise NotImplementedError

    return generator
