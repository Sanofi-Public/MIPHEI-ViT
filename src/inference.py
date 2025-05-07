"""
Test scripts for image to image segmentation.
"""
import logging
logging.getLogger('pyvips').setLevel(logging.WARNING)

import os
os.environ['VIPS_CONCURRENCY'] = '1'
import pyvips

from omegaconf import OmegaConf
import albumentations as A
import json
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
import torch
from slidevips.torch_datasets import SlideDataset
from timm.layers import resample_abs_pos_embed

from .dataset import NormalizationLayer, get_effective_width_height, TileSlideDataset, get_width_height
from .models import ModelModule, DiscriminatorPatch
from .callbacks import SavePredictionsCallback
from .generators import get_generator
from timm.layers import resample_patch_embed, resize_rel_pos_bias_table


def validate_load_info(load_info):
    """
    Validates the result of model.load_state_dict(..., strict=False).

    Raises:
        ValueError if unexpected keys are found,
        or if missing keys are not related to the allowed encoder modules.
    """
    # 1. Raise if any unexpected keys
    if load_info.unexpected_keys:
        raise ValueError(f"Unexpected keys in state_dict: {load_info.unexpected_keys}")

    # 2. Raise if any missing keys are not part of allowed encoder modules
    for key in load_info.missing_keys:
        if ".lora" in key:
            raise ValueError(f"Missing LoRA checkpoint in state_dict: {key}")
        elif not any(part in key for part in ["encoder.vit.", "encoder.model."]):
            raise ValueError(f"Missing key in state_dict: {key}")


def resize_embed_hemit_statedict(state_dict, model):
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'attn_mask')]):
                continue

        if 'swinT.patch_embed.proj.weight' in k:
            _, _, H, W = model.swinT.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        new_state_dict[k] = v

    return new_state_dict


def get_generator_state_dict(state_dict):
    generator_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            generator_state_dict[k.replace("generator.", "")] = v
    return generator_state_dict


def inference_model(cfg, checkpoint_dir, output_dir):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    pyvips.cache_set_max(0)
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

    with open(cfg.data.channel_stats_path, "r") as f:
        channel_stats = json.load(f)

    width, height = get_width_height(test_dataframe)
    width, height = get_effective_width_height(width, height, train=True)

    spatial_augmentations = A.Compose([
        A.CenterCrop(width=width, height=height),
    ], additional_targets={"image_target": "image", "nuclei": "image"})
    nc_out = len(cfg.data.targ_channel_names)
    nc_in = 3
    log.info("{} width / {} height".format(width, height))
    log.info("{} inputs channels / {} output channels".format(nc_in, nc_out))

    preprocess_input_fn = NormalizationLayer(channel_stats["RGB"], mode="he")

    if from_slide:
        dataset = SlideDataset(slide_dataframe=slide_dataframe, dataframe=test_dataframe)
    else:
        dataset = TileSlideDataset(
            dataframe=test_dataframe,preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=spatial_augmentations)

    num_workers = os.cpu_count() - 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train.batch_size, pin_memory=device!="cpu",
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
        #generator = torch.compile(generator)
        #jit_compile = True
        jit_compile = False


    discriminator = None
    #foreground_loss = CombinedBCEAndDiceLoss(1.)
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
