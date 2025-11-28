# @GonzaloMartinGarcia
# This file contains Marigold's exponential LR scheduler. 
# https://github.com/prs-eth/Marigold/blob/main/src/util/lr_scheduler.py

# Author: Bingxin Ke
# Last modified: 2024-02-22

import numpy as np
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F


class IterExponential:
    
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


# Function is based on an early commit of the Marigold GitHub repository.
def replace_unet_conv_in(unet, repeat=2):
    _weight = unet.conv_in.weight.clone() 
    _bias = unet.conv_in.bias.clone() 
    _weight = _weight.repeat((1, repeat, 1, 1))  
    # scale the activation magnitude
    _weight /= repeat
    _bias /= repeat
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4*repeat, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # replace config
    unet.config.in_channels = 4*repeat
    unet.config['in_channels'] = 4*repeat
    return


def add_marker_embedding_to_unet(unet, num_channels):
    unet._set_class_embedding(
        class_embed_type="projection",
        time_embed_dim=unet.time_embedding.linear_2.out_features,
        projection_class_embeddings_input_dim=num_channels * 2,
        act_fn=None,  # dummy
        timestep_input_dim=None,  # dummy
        num_class_embeds=None
    )
    unet.config.class_embed_type = "projection"
    unet.config.projection_class_embeddings_input_dim = num_channels * 2


def pixel_mix_loss(pred_img, gt_img, lam, reduction: str = "mean"):
    """
    L_FT = (1 - λ) * L1 + λ * L2 between reconstructed `pred_img` and target `gt_img`.

    Args:
        pred_img: BCHW tensor (e.g., VAE decode output)
        gt_img:   BCHW tensor (same shape/range as pred_img)
        lam:      λ in [0,1]
        reduction: 'mean' | 'sum' | 'none'
    """
    if lam == 0.:
        return F.l1_loss(pred_img, gt_img, reduction=reduction)
    elif lam == 1.:
        return F.mse_loss(pred_img, gt_img, reduction=reduction)
    else:
        return (1 - lam) * F.l1_loss(pred_img, gt_img, reduction=reduction) \
            + lam * F.mse_loss(pred_img, gt_img, reduction=reduction)
