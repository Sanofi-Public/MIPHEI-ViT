"""
MIPHEI-ViT model inspired by ViTMatte model.

This is a modified version of the original code from the repository:
https://github.com/hustvl/ViTMatte/blob/main/modeling/meta_arch/vitmatte.py
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import SwinTransformer, VisionTransformer

from .foundation_models import FOUNDATION_MODEL_REGISTRY
from .lora import apply_lora
from .smp_unet import SegmentationHead, initialize_decoder_head


def get_mipheivit(encoder_name: str, img_size: int, num_classes: int, use_lora: bool = False,
                  ckpt_path: Optional[str] = None, drop_path_rate: float = 0) -> nn.Module:
    """
    Construct and return a MIPHEIViT model with the specified configuration.

    Args:
        encoder_name (str): Name of the foundation model to use from the FOUNDATION_MODEL_REGISTRY.
        img_size (int or tuple): Input image size for the encoder.
        num_classes (int): Number of output classes for the decoder.
        use_lora (bool, optional): Whether to apply LoRA (Low-Rank Adaptation) with rank 8 and
            alpha 1 to the encoder. Defaults to False.
        ckpt_path (str or None, optional): Path to a checkpoint file to initialize the encoder
            weights. Defaults to None.
        drop_path_rate (float, optional): Drop path rate for stochastic depth regularization in the
            encoder. Defaults to 0.

    Returns:
        MIPHEIViT: An instance of the MIPHEIViT model.
    """
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")

    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    encoder = Encoder(vit)
    decoder = Detail_Capture(emb_chans=encoder.embed_dim, out_chans=num_classes,
                             use_attention=True, activation=nn.Tanh())
    model = MIPHEIViT(encoder=encoder, decoder=decoder)
    return model


class Basic_Conv3x3(nn.Module):
    """
    Basic convolutional block with Conv3x3, BatchNorm2d, and ReLU layers.

    This module applies a 2D convolution with a 3x3 kernel, followed by batch normalization and
    a ReLU activation.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 2.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
    Attributes:
        conv (nn.Conv2d): 2D convolutional layer with a 3x3 kernel.
        bn (nn.BatchNorm2d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution block to the input tensor."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvStream(nn.Module):
    """
    A simple convolutional stream composed of sequential 3x3 convolutional layers for extracting \
        detailed / high resolution features.

    Args:
        in_chans (int, optional): Number of input channels. Defaults to 3.
        out_chans (List[int], optional): List of output channels for each convolutional layer.
            Defaults to [48, 96, 192].
    Attributes:
        convs (nn.ModuleList): List of sequential Basic_Conv3x3 layers.
        conv_chans (List[int]): List of channel sizes for each convolutional layer, including input
            channels.
    Methods:
        forward(x):
            Applies the convolutional stream to the input tensor and returns a dictionary of
                intermediate feature maps.

            Args:
                x (torch.Tensor): Input tensor of shape (B, C, H, W).

            Returns:
                Dict[str, torch.Tensor]: Dictionary containing the input and outputs of each
                    convolutional layer. Keys are 'D0' (input), 'D1', 'D2', ..., corresponding to
                    each layer's output.
    """

    def __init__(
        self,
        in_chans: int = 3,
        out_chans: List[int] = [48, 96, 192],
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

    def forward(self, x: torch.Tensor) -> dict:
        """Apply convstream to extract skip connections from images x."""
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x

        return out_dict


class Fusion_Block(nn.Module):
    """
    Fusion block that merges ViT features with convolutional skip connections.

    This module upsamples the input feature map, concatenates it with a corresponding
    skip connection, and applies a convolution to fuse the features. It mimics the behavior
    of decoder blocks in U-Net-like architectures.

    Args:
        in_chans (int): Number of input channels after concatenation.
        out_chans (int): Number of output channels for the fused features.
    Attributes:
        conv (nn.Module): A 3x3 convolutional layer for feature fusion.
    Methods:
        forward(x, D):
            Args:
                x (torch.Tensor): Decoder input feature map to be upsampled.
                D (torch.Tensor): Skip connection feature map from earlier stage.

            Returns:
                torch.Tensor: Fused feature map.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """Forward pass of fusion block between features and skip connection."""
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out


class Encoder(nn.Module):
    """
    Encoder module that wraps a Vision Transformer or Swin Transformer for feature extraction.

    This class adapts the output of a VisionTransformer or SwinTransformer to output the spatially
    reconstructed token features. Prefix token are discarded and if the patch size is not 16x16,
    the output is resized to a target scale using bicubic interpolation, compatible with UNet.

    Args:
        vit (VisionTransformer or SwinTransformer): The transformer model to be used as the encoder.
    Attributes:
        vit (VisionTransformer or SwinTransformer): The transformer model to be wrapped.
        is_swint (bool): Flag indicating if the model is a SwinTransformer.
        grid_size (tuple): The spatial grid size of the patch embeddings.
        num_prefix_tokens (int): Number of prefix tokens in the transformer (ViT only).
        embed_dim (int): Embedding dimension of the transformer output.
        scale_factor (tuple or None): Scaling factor for resizing the feature map. Set to None if
            the patch size is 16x16.
    Raises:
        ValueError: If the provided model is not a VisionTransformer or SwinTransformer.
    Methods:
        forward(x):
            Extracts features from the input tensor using the transformer model and returns a
            spatial feature map using token features only, optionally resizing it to a target scale.
    """

    def __init__(self, vit: nn.Module):
        super().__init__()
        if not isinstance(vit, (VisionTransformer, SwinTransformer)):
            raise ValueError(
                f"Model should be a VisionTransformer or SwinTransformer, got {type(vit)}")
        self.vit = vit

        self.is_swint = isinstance(vit, SwinTransformer)
        self.grid_size = self.vit.patch_embed.grid_size
        if self.is_swint:
            self.num_prefix_tokens = 0
            self.embed_dim = self.vit.embed_dim * 2 ** (self.vit.num_layers - 1)
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
                self.scale_factor = (target_grid_size[0] / self.grid_size[0],
                                     target_grid_size[1] / self.grid_size[1])
            else:
                self.scale_factor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial feature extraction."""
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
    Decoder module that fuses ViT features with high-resolution convolutional features (UNet like).

    Inspired by the Detail Capture Module from ViTMatte, this decoder fuses spatial ViT embeddings
    with high-resolution features from a convolutional stream using stacked UNet-like fusion
    blocks. Each output channel is then generated by a separate head.

    Args:
        emb_chans (int): Number of channels in the transformer embedding features.
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        out_chans (int, optional): Number of output channels (number of segmentation heads).
            Defaults to 1.
        convstream_out (list of int, optional): Output channels for each stage of the convolutional
            stream. Defaults to [48, 96, 192].
        fusion_out (list of int, optional): Output channels for each fusion block. Must be one more
            than convstream_out. Defaults to [256, 128, 64, 32].
        use_attention (bool, optional): Whether to use attention in the segmentation heads.
            Defaults to True.
        activation (nn.Module, optional): Activation function to use in the segmentation heads.
            Defaults to nn.Identity().
    Attributes:
        convstream (ConvStream): Convolutional stream for extracting detail features from input
            images.
        conv_chans (list of int): Output channels of the convolutional stream.
        num_heads (int): Number of segmentation heads (output channels).
        fusion_blks (nn.ModuleList): List of fusion blocks for combining transformer and detail
            features.
        fus_channs (list of int): Channel sizes for each fusion stage.
        segmentation_head_{idx} (SegmentationHead): Segmentation head modules for each output
            channel.
    Methods:
        forward(features, images):
            Forward pass of the module. Fuses transformer features with detail features and
            produces output masks/images.
    """

    def __init__(
        self,
        emb_chans: int,
        in_chans: int = 3,
        out_chans: int = 1,
        convstream_out: List[int] = [48, 96, 192],
        fusion_out: List[int] = [256, 128, 64, 32],
        use_attention: bool = True,
        activation=nn.Identity()
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans=in_chans, out_chans=convstream_out)
        self.conv_chans = self.convstream.conv_chans
        self.num_heads = out_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, emb_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans=self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans=self.fus_channs[i+1],
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

    def forward(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Forward pass of decoder using detail capture."""
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


class MIPHEIViT(nn.Module):
    """
    Model for image to image translation using a transformer backbone.

    MIPHEI-ViT is a hybrid UNet-style encoder-decoder model for image-to-image translation. It uses
    a (pretrained) transformer encoder (ViT or Swin) and a convolutional stream to extract
    high-resolution features, which are fused in the decoder. Inspired by ViTMatte, it is adapted
    for multiplex immunofluorescence prediction from H&E images.
    Paper: https://arxiv.org/abs/2505.10294

    Args:
        encoder (nn.Module): The transformer-based encoder returning spatial feature maps.
        decoder (nn.Module): The decoder that fuses transformer features with convolutional skip
            connections.
    Attributes:
        encoder (nn.Module): Vision transformer encoder (modified for spatial token output).
        decoder (nn.Module): UNet-like decoder with fusion blocks.
    Methods:
        forward(x):
            Performs a forward pass through encoder and decoder.
        initialize():
            Initializes decoder weights using Pix2Pix-like initialization.
        set_input_size(img_size):
            Configures input size for ViT and updates grid size.
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ):
        super(MIPHEIViT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of MIPHEI using input tensor x."""
        features = self.encoder(x)
        outputs = self.decoder(features, x)
        return outputs

    def initialize(self) -> None:
        """Initialize the decoder head for the model."""
        initialize_decoder_head(self.decoder)

    def set_input_size(self, img_size: Tuple[int]) -> None:
        """Set the input image size, ensuring both dimensions are powers of 2 and at least 128."""
        if any((s & (s - 1)) != 0 or s == 0 for s in img_size):
            raise ValueError("Both height and width in img_size must be powers of 2")
        if any(s < 128 for s in img_size):
            raise ValueError("Height and width must be greater or equal to 128")
        self.encoder.vit.set_input_size(img_size=img_size)
        self.encoder.grid_size = self.encoder.vit.patch_embed.grid_size
