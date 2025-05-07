"""
MIPHEI-vit model inspired by ViTMatte model.
This is a modified version of the original code from the repository:
https://github.com/hustvl/ViTMatte/blob/main/modeling/meta_arch/vitmatte.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.layers import resample_abs_pos_embed
from timm.models import VisionTransformer, SwinTransformer

from .unet import SegmentationHead, initialize_decoder_head
from .lora import apply_lora
from .foundation_models import FOUNDATION_MODEL_REGISTRY


class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_)
            )
    
    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        
        return out_dict


class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) ## Nearest ?
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out


class ViTMatte(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.initialize()

    def forward(self, x):

        features = self.encoder(x)
        outputs = self.decoder(features, x)
        return outputs

    def initialize(self):
        initialize_decoder_head(self.decoder)

    def set_input_size(self, img_size):
        if any((s & (s - 1)) != 0 or s == 0 for s in img_size):
            raise ValueError("Both height and width in img_size must be powers of 2")
        if any(s < 128 for s in img_size):
            raise ValueError("Height and width must be greater or equal to 128")
        self.encoder.vit.set_input_size(img_size=img_size)
        self.encoder.grid_size = self.encoder.vit.patch_embed.grid_size


class Encoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        if not isinstance(vit, (VisionTransformer, SwinTransformer)):
            raise ValueError(f"Model should be a VisionTransformer or SwinTransformer, got {type(vit)}")
        self.vit = vit

        self.is_swint = isinstance(vit, SwinTransformer)
        self.grid_size = self.vit.patch_embed.grid_size
        if self.is_swint:
            self.num_prefix_tokens = 0
            self.embed_dim = self.vit.embed_dim * 2 ** (self.vit.num_layers -1)
        else:
            self.num_prefix_tokens = self.vit.num_prefix_tokens
            self.embed_dim = self.vit.embed_dim
        patch_size = self.vit.patch_embed.patch_size
        img_size = self.vit.patch_embed.img_size
        assert img_size[0] % 16 == 0
        assert img_size[1] % 16 == 0

        if self.is_swint:
            self.scale_factor = (2., 2.)
        else:
            if patch_size != (16, 16):
                target_grid_size = (img_size[0] / 16, img_size[1] / 16)
                self.scale_factor = (target_grid_size[0] / self.grid_size[0], target_grid_size[1] / self.grid_size[1])
            else:
                self.scale_factor = None

    def forward(self, x):
        features = self.vit(x)
        if self.is_swint:
            features = features.permute(0, 3, 1, 2)
        else:
            features = features[:, self.num_prefix_tokens:]
            features = features.permute(0, 2, 1)
            features = features.view((-1, self.embed_dim, *self.grid_size))
        if self.scale_factor is not None:
            features = F.interpolate(features, scale_factor=self.scale_factor, mode="bicubic")
        return features


class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        emb_chans,
        in_chans=3,
        out_chans=1,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
        use_attention=True,
        activation=torch.nn.Identity()
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans=in_chans)
        self.conv_chans = self.convstream.conv_chans
        self.num_heads = out_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, emb_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        for idx in range(self.num_heads):
            setattr(self, f'segmentation_head_{idx}', SegmentationHead(
                in_channels=fusion_out[-1],
                out_channels=1,
                activation=activation,
                kernel_size=3,
                use_attention=use_attention
            ))

    def forward(self, features, images):
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blks)):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = self.fusion_blks[i](features, detail_features[d_name_])
        
        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(features)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)

        return outputs



def get_vitmatte(encoder_name, img_size, num_classes, use_lora=False, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")

    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    encoder = Encoder(vit)
    decoder = Detail_Capture(emb_chans=encoder.embed_dim, out_chans=num_classes, use_attention=True, activation=nn.Tanh())
    model = ViTMatte(encoder=encoder, decoder=decoder)
    return model
