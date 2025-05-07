"""
Utility functions for the project.
"""

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from wandb import Artifact


class MeanCellExtrator(nn.Module):
    def __init__(self, scale_factor=1.):
        super(MeanCellExtrator, self).__init__()
        if not (0. < scale_factor <= 1):
            raise ValueError("scale_factor should be between 0 and 1")
        self.scale_factor = scale_factor

    def forward(self, pred, target, nuclei):
        """
        Compute L1 loss between mean intensities extracted from prediction and target for regions defined by nuclei.

        Args:
            pred (torch.Tensor): Predicted tensor of shape [B, C, H, W].
            target (torch.Tensor): Target tensor of shape [B, C, H, W].
            nuclei (torch.Tensor): Nuclei label tensor of shape [B, 1, H, W].

        Returns:
            torch.Tensor: L1 loss between extracted mean intensities.
        """
        # Downsample pred, target, and nuclei
        if target==None:
            target = torch.zeros_like(pred)
        if nuclei.ndim == 3:
            nuclei = torch.unsqueeze(nuclei, dim=1).long()
        if self.scale_factor < 1.:
            pred = F.interpolate(pred, scale_factor=self.scale_factor, mode='area')
            target = F.interpolate(target, scale_factor=self.scale_factor, mode='area')
            nuclei = F.interpolate(nuclei.float(), scale_factor=self.scale_factor, mode='nearest-exact').long()

        # Extract mean intensities for both pred and target
        pred_means, target_means, cell_ids = self.extract_mean(pred, target, nuclei)
        return pred_means, target_means, cell_ids

    def extract_mean(self, pred, target, nuclei):
        """
        Extract mean intensities for prediction and target using shared nuclei computations.

        Args:
            pred (torch.Tensor): Predicted tensor of shape [B, C, H, W].
            target (torch.Tensor): Target tensor of shape [B, C, H, W].
            nuclei (torch.Tensor): Nuclei label tensor of shape [B, 1, H, W].

        Returns:
            torch.Tensor: Mean intensities for prediction, shape [num_labels, C].
            torch.Tensor: Mean intensities for target, shape [num_labels, C].
            list[torch.Tensor]: Unique cell IDs for each batch element.
        """
        batch_size, num_channels, height, width = pred.shape

        # Prepare outputs
        all_pred_means = []
        all_target_means = []
        all_cell_ids = []

        for b in range(batch_size):
            # Process each batch independently
            nuclei_b = nuclei[b, 0]  # Shape: [H, W]
            pred_b = pred[b]  # Shape: [C, H, W]
            target_b = target[b]  # Shape: [C, H, W]

            # Create a binary mask for non-background pixels
            nuclei_binary = nuclei_b > 0

            # Apply the binary mask to nuclei
            nuclei_flat = nuclei_b[nuclei_binary]  # Shape: [num_valid_pixels]
            if nuclei_flat.numel() == 0:  # No valid regions
                all_pred_means.append(torch.zeros((0, num_channels), dtype=pred.dtype, device=pred.device))
                all_target_means.append(torch.zeros((0, num_channels), dtype=target.dtype, device=target.device))
                all_cell_ids.append(torch.empty(0, dtype=nuclei_b.dtype, device=nuclei_b.device))
                continue

            # Get unique labels and their indices
            unique_labels, inverse_indices = torch.unique(nuclei_flat, return_inverse=True)

            # Apply the mask to pred and target, flatten and preserve the channel dimension
            pred_flat = pred_b.permute(1, 2, 0)[nuclei_binary]  # Shape: [num_valid_pixels, C]
            target_flat = target_b.permute(1, 2, 0)[nuclei_binary]  # Shape: [num_valid_pixels, C]

            # Compute sums per region and channel for pred and target
            pred_sums = torch.zeros((unique_labels.shape[0], num_channels), dtype=pred.dtype, device=pred.device).scatter_add_(
                0, inverse_indices.unsqueeze(1).expand(-1, num_channels), pred_flat
            )
            target_sums = torch.zeros((unique_labels.shape[0], num_channels), dtype=target.dtype, device=target.device).scatter_add_(
                0, inverse_indices.unsqueeze(1).expand(-1, num_channels), target_flat
            )

            # Compute counts per region
            region_counts = torch.zeros(unique_labels.shape[0], dtype=torch.float32, device=nuclei.device).scatter_add_(
                0, inverse_indices, torch.ones_like(nuclei_flat, dtype=torch.float32)
            )

            # Compute mean intensities
            pred_means = pred_sums / region_counts.unsqueeze(1)
            target_means = target_sums / region_counts.unsqueeze(1)

            # Append results
            all_pred_means.append(pred_means)
            all_target_means.append(target_means)
            all_cell_ids.append(unique_labels)

        # Concatenate results across batches
        pred_means = torch.cat(all_pred_means, dim=0)
        target_means = torch.cat(all_target_means, dim=0)
        cell_ids = torch.cat(all_cell_ids, dim=0)

        return pred_means.to(pred.dtype), target_means.to(pred.dtype), cell_ids


def save_torch_weights(model: pl.LightningModule, torch_weights: str) -> None:
    """
    Save the weights of a Lightning model to a Torch-compatible file.

    Args:
        model (TO DO): The Lightning model whose weights will be saved.
        torch_weights (str): The file path where the Torch-compatible weights will be saved.
    """
    lightning_state_dict = model.generator.state_dict()

    torch_state_dict = {}
    for layer_name, weights in lightning_state_dict.items():
        layer_name_split = layer_name.split(".")
        if layer_name_split[1] == "_orig_mod":
            new_layer_name = ".".join(layer_name.split(".")[2:])
        else:
            new_layer_name = ".".join(layer_name.split(".")[1:])
        torch_state_dict[new_layer_name] = weights

    torch.save(torch_state_dict, torch_weights)


def load_normalization_stats(stats_in_path: str, stats_out_path: str
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load normalization statistics from the given paths.

    Args:
        stats_in_path (str): Path to the input statistics file.
        stats_out_path (str): Path to the output statistics file.

    Returns:
        tuple: A tuple containing the mean and standard deviation for the input image and target.
               The tuple is in the order (mean_image, std_image, mean_target, std_target).
    """
    stats_image = np.load(stats_in_path)
    stats_target = np.load(stats_out_path)
    mean_image = torch.from_numpy(stats_image[0]).reshape((1, -1, 1, 1))
    std_image = torch.from_numpy(stats_image[1]).reshape((1, -1, 1, 1))
    min_target = torch.from_numpy(stats_target[0]).reshape((1, -1, 1, 1))
    max_target = torch.from_numpy(stats_target[1]).reshape((1, -1, 1, 1))
    return mean_image, std_image, min_target, max_target


def get_dim_images(val_dataset) -> Tuple[int, int]:
    """
    Get the number of input and output channels using the validation dataset.
        These values are useful for the augmentations.

    Args:
        val_dataset: The validation dataset.

    Returns:
        A tuple containing the number of input channels and the number of output channels.
    """
    val_batch = val_dataset[0]
    nc_in, width, height = val_batch[0].shape
    nc_out, width_out, height_out = val_batch[1].shape
    assert width == width_out and height == height_out
    val_dataset.reset()
    return nc_in, nc_out, width, height


def wandb_log_artifact(logger, artifact_name: str, artifact_type: str, file_path: str):
    run = logger.experiment
    assert run is not None
    artifact_name = f"run_{run.id}_{artifact_name}"
    artifact = Artifact(artifact_name, type=artifact_type)
    artifact.add_file(file_path)
    logger.experiment.log_artifact(artifact)


def update_wandb_note(wandb_note):
    hydra_name = HydraConfig.get().job['override_dirname']
    wandb_note = wandb_note + " /" + hydra_name
    return wandb_note


def get_foreground_weight(channel_names, train_dataframe):
    columns = [f"{channel_name}_prop" for channel_name in channel_names]
    foreground_prop = train_dataframe[columns].mean(axis=0).values
    foreground_weight = 1 - foreground_prop
    return np.maximum(foreground_weight / (1 - foreground_weight), 1.) #########
    #return foreground_weight ########


def get_foreground_thresh(stats_list_if, preprocess_target_fn):
    stats_list_if = stats_list_if.copy()
    min_ = np.array([stats["min"] for stats in stats_list_if])
    tresh = preprocess_target_fn(min_).reshape((1, 1, 1, -1))
    return torch.from_numpy(tresh).float()


def pix2pix_lr_scheduler(total_iters, warmup_iters, decay_start_iter):
    def lr_lambda(step):
        if step < warmup_iters:
            # Warmup Phase: linearly increase from 0 to initial_lr
            return step / warmup_iters
        elif step < decay_start_iter:
            # Constant Phase: hold initial_lr until decay_start_iter
            return 1.0
        else:
            # Decay Phase: linearly decay to zero from decay_start_iter to total_iters
            decay_steps = total_iters - decay_start_iter
            decay_factor = (step - decay_start_iter) / decay_steps
            return max(0.0, 1.0 - decay_factor)
    return lr_lambda


class LayerDecayOptimizerConstructor:
    def __init__(self, base_lr, base_wd, paramwise_cfg=None):
        self.base_lr = base_lr
        self.base_wd = base_wd
        self.paramwise_cfg = paramwise_cfg if paramwise_cfg is not None else {}

    @staticmethod
    def get_num_layer_for_vit(self, var_name, num_max_layer):
        # Function remains the same, it's logic-based and framework-independent
        if var_name in ('encoder.vit_adapter.cls_token', 'encoder.vit_adapter.mask_token',
                        'encoder.vit_adapter.pos_embed', 'encoder.vit_adapter.visual_embed'):
            return 0
        elif var_name.startswith('encoder.vit_adapter.patch_embed') or \
            var_name.startswith('encoder.vit_adapter.visual_embed'):
            return 0
        elif var_name.startswith('encoder.vit_adapter.blocks') or \
            var_name.startswith('encoder.vit_adapter.layers'):
            layer_id = int(var_name.split('.')[3])
            return layer_id + 1
        else:
            return num_max_layer - 1

    def construct(self, model):
        params = []
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        weight_decay = self.base_wd

        parameter_groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = self.get_num_layer_for_vit(name, num_layers)
            group_name = f'layer_{layer_id}_{group_name}'
            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }
            parameter_groups[group_name]['params'].append(param)

        params.extend(parameter_groups.values())
        return params


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("encoder.model"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)
