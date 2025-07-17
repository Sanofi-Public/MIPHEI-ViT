"""
Custom implementation of UNETR / CellViT model.

Inspired by https://github.com/TIO-IKIM/CellViT/blob/main/models/segmentation/cell_segmentation/cellvit.py#L26
Custom implementation allows FeatureUpsampler for ViT and SwinViT making it
compatible with any plain ViT/SwinTransformer foundation model and some
adaptations for image translations (tanh etc.)
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from timm.models import SwinTransformer, VisionTransformer

from .foundation_models import FOUNDATION_MODEL_REGISTRY
from .lora import apply_lora
from .smp_unet import SegmentationHead, initialize_decoder_head


class Conv2DBlock(nn.Module):
    """
    Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout.

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Conv block module to the input tensor x."""
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """
    Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation,\
    ReLU activation and dropout.

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Deconv block module to the input tensor x."""
        return self.block(x)


class Resnet50LunitSwav(nn.Module):
    """
    Adapter class to use the Lunit Resnet50 backbone (restnet50_lunit_swav) as a UNet encoder.

    This wraps the Lunit Resnet50 model and adds a convolutional stem for compatibility with the
    UNet-style decoder to extract high level features.

    Args:
        ckpt_path (str): Path to the checkpoint file for loading pre-trained weights into
            the backbone model. Defaults to None.
        drop_rate (float): Dropout rate applied in convolutional blocks and the backbone.
            Defaults to 0.
    """

    def __init__(self, ckpt_path: Optional[str] = None, drop_rate: float = 0.):
        super().__init__()
        self.model = FOUNDATION_MODEL_REGISTRY["restnet50_lunit_swav"](
                ckpt_path=ckpt_path, drop_rate=drop_rate)
        self.convsteam = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=drop_rate),
            Conv2DBlock(32, 64, 3, dropout=drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.out_channels = [64, 64, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass that returns pyramidal features for UNet."""
        features_convsteam = self.convsteam(x)
        features = self.model.forward_intermediates(
            x,
            indices=(0, 1, 2, 3),
            intermediates_only=True
            )
        return [features_convsteam] + features


class ViTFeatureUpsampler(nn.Module):
    """
    Upsamples features from a Vision Transformer (ViT) backbone for use in decoder architectures.

    This module processes input images and a list of feature maps from a ViT encoder, applying
    a series of convolutional and deconvolutional blocks to upsample ViT features. It is designed
    to facilitate skip connections and multi-scale feature fusion in segmentation or image-to-image
    translation tasks.

    Args:
        embed_dim (int): The embedding dimension of the ViT features.
        drop_rate (float): Dropout rate for regularization.
        scale_factor (float or None, optional): Applies bilinear upsampling to match UNet feature
            size requirements. Defaults to None (no upsampling).
    Attributes:
        embed_dim (int): The embedding dimension of the ViT features.
        drop_rate (float): Dropout rate applied in convolutional and deconvolutional blocks.
        scale_factor (float or None): Upsampling scale factor for feature maps.
            If None, no upsampling is applied.
        skip_dim_11 (int): Intermediate channel dimension for skip connection 1.
        skip_dim_12 (int): Intermediate channel dimension for skip connection 1.
        bottleneck_dim (int): Channel dimension for the bottleneck upsampling block.
        convsteam (nn.Sequential): Convolutional stem for high resolution feature extraction.
        upsampler0 (nn.Sequential): Upsampling path for the first skip connection.
        upsampler1 (nn.Sequential): Upsampling path for the second skip connection.
        upsampler2 (nn.Sequential): Upsampling path for the third skip connection.
        upsampler3 (nn.Sequential): Upsampling path for the fourth skip connection.
        out_channels (list): List of output channel dimensions for each upsampling path.

    Methods:
        forward(x, features):
            Processes the input image and feature maps, returning a list of upsampled feature maps
                at multiple scales.
    """

    def __init__(self, embed_dim: int, drop_rate: float, scale_factor: float = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.scale_factor = scale_factor
        if embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        self.convsteam = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64

        self.upsampler0 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest")
            if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.upsampler1 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest")
            if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.upsampler2 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest")
            if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3
        self.upsampler3 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest")
            if self.scale_factor else nn.Identity(),
        )  # skip connection 3
        self.out_channels = [
            self.convsteam[-1].out_channels,
            self.upsampler0[-1].out_channels,
            self.upsampler1[-1].out_channels,
            self.upsampler2[-1].out_channels,
            self.embed_dim,
        ]
        initialize_decoder_head(self)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass that upsample ViT features."""
        features_convsteam = self.convsteam(x)
        features0 = features[0]
        features0 = self.upsampler0(features0)
        features1 = features[1]
        features1 = self.upsampler1(features1)
        features2 = features[2]
        features2 = self.upsampler2(features2)
        features3 = features[3]
        features3 = self.upsampler3(features3)
        return [features_convsteam, features0, features1, features2, features3]


class SwinViTFeatureUpsampler(nn.Module):
    """
    Upsamples features from a Swin Transformer backbone for use in decoder architectures.

    Similarly to similar to `ViTFeatureUpsampler`, this module processes input images and a list of
    feature maps from a Swin Transformer encoder, applying a series of convolutional and
    deconvolutional blocks to upsample Swin Transformer features. It is designed to facilitate skip
    connections and multi-scale feature fusion in segmentation or image-to-image translation tasks.
    The difference with `ViTFeatureUpsampler` is that Swin Transformer models already return low
    resolution pyramidal features, so upsampling is different.

    Args:
        embed_dims (List[int]): List of embedding dimensions for each feature scale from
            the Swin Transformer.
        drop_rate (float): Dropout rate applied in convolutional and deconvolutional blocks.
    Attributes:
        embed_dims (List[int]): List of embedding dimensions for each feature scale from
            the Swin Transformer.
        drop_rate (float): Dropout rate applied in convolutional and deconvolutional blocks.
        bottleneck_dim (int): Bottleneck dimension for the deepest feature map, determined by
            the last embed_dim.
        convsteam (nn.Sequential): Sequential block of convolutional layers for high resolution
            feature extraction.
        upsampler0 (nn.Sequential): Upsampling block for the first feature scale.
        upsampler1 (nn.Sequential): Upsampling block for the second feature scale.
        upsampler2 (nn.Sequential): Upsampling block for the third feature scale.
        upsampler3 (nn.Sequential): Upsampling block for the fourth feature scale.
        out_channels (List[int]): List of output channels for each upsampled feature map.
    Methods:
        forward(x, features):
            Processes the input image and feature maps, returning a list of upsampled feature maps
                at multiple scales.
    """

    def __init__(self, embed_dims: List[int], drop_rate: float):
        super().__init__()
        self.embed_dims = embed_dims
        self.drop_rate = drop_rate
        if self.embed_dims[-1] < 512:
            self.bottleneck_dim = 312
        else:
            self.bottleneck_dim = 512

        self.convsteam = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64

        self.upsampler0 = nn.Sequential(
            Deconv2DBlock(self.embed_dims[0], 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.upsampler1 = nn.Sequential(
            Deconv2DBlock(self.embed_dims[1], 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.upsampler2 = nn.Sequential(
            Deconv2DBlock(self.embed_dims[2], self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3
        self.upsampler3 = nn.Sequential(
            Deconv2DBlock(self.embed_dims[3], self.embed_dims[3], dropout=self.drop_rate)
        )  # skip connection 3
        self.out_channels = [
            self.convsteam[-1].out_channels,
            self.upsampler0[-1].out_channels,
            self.upsampler1[-1].out_channels,
            self.upsampler2[-1].out_channels,
            self.embed_dims[-1],
        ]
        initialize_decoder_head(self)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass that extracts high level features and upsample vit features."""
        features_convsteam = self.convsteam(x)
        features0 = features[0]
        features0 = self.upsampler0(features0)
        features1 = features[1]
        features1 = self.upsampler1(features1)
        features2 = features[2]
        features2 = self.upsampler2(features2)
        features3 = features[3]
        features3 = self.upsampler3(features3)
        return [features_convsteam, features0, features1, features2, features3]


class ViTPyramidEncoder(nn.Module):
    """
    ViTPyramidEncoder is a feature encoder designed to extract and upsample pyramidal features from\
    Vision Transformer (ViT) or Swin Transformer, following the UNetR approach.

    It enables hierarchical feature extraction at multiple scales for downstream tasks such as
    segmentation.

    Args:
        img_size (int): Input image size (assumed square).
        encoder_name (str): Name of the transformer encoder to use, must be registered in
            FOUNDATION_MODEL_REGISTRY.
        ckpt_path (str, optional): Path to a pretrained checkpoint for the encoder.
            Defaults to None.
        drop_path_rate (float, optional): Drop path rate for stochastic depth regularization.
            Defaults to 0.
        use_lora (bool, optional): Whether to apply LoRA (Low-Rank Adaptation) to the encoder.
            If True, applies LoRA with rank 8 and alpha 1. to the encoder. Defaults to False.
    Attributes:
        model (nn.Module): The underlying VisionTransformer or SwinTransformer model.
        extract_layers (list of int): Indices of layers from which to extract features for the
            pyramid.
        patch_size (int): Patch size used by the encoder.
        drop_rate (float): Dropout path rate.
        feature_upsampler (nn.Module): Module to upsample features to a common spatial resolution.
        out_channels (list of int): Number of output channels for each pyramid level.
    Methods:
        forward_features(x):
            Extracts intermediate features from the specified layers of the encoder.

        forward(x):
            Returns upsampled pyramidal features suitable for use in UNetR-style decoders.
    """

    def __init__(self, img_size: int, encoder_name: str, ckpt_path: Optional[str] = None,
                 drop_path_rate: float = 0., use_lora: bool = False):
        super().__init__()
        try:
            model = FOUNDATION_MODEL_REGISTRY[encoder_name](
                img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate)
        except KeyError:
            raise NotImplementedError(
                f"Unknown model: try ones in {list(FOUNDATION_MODEL_REGISTRY.keys())}")
        if not isinstance(model, (VisionTransformer, SwinTransformer)):
            raise ValueError(
                f"Model should be a VisionTransformer or SwinTransformer, got {type(model)}")
        self.model = model
        if use_lora:
            apply_lora(self.model, rank=8, alpha=1.)

        is_vit = isinstance(self.model, VisionTransformer)
        depth = len(model.blocks) if is_vit else len(model.layers)
        if depth == 4:
            self.extract_layers = [0, 1, 2, 3]
        elif depth > 4:
            self.extract_layers = np.round(
                np.linspace(depth // 4, depth - 1, 4)).astype(int).tolist()
        else:
            raise ValueError("Vit Should have a depth higher than 3")

        self.patch_size = 16
        self.drop_rate = drop_path_rate

        assert img_size % self.patch_size == 0

        if is_vit:
            real_patch_size = self.model.patch_embed.patch_size[0]
            # UNet needs spatial resolution divisible by 16 if not features will be upsampled using
            # bilinear interpolation
            if real_patch_size != 16:
                scale_factor = int((img_size / 16)) / int(img_size / real_patch_size)
            else:
                scale_factor = None
            self.feature_upsampler = ViTFeatureUpsampler(
                self.model.embed_dim, scale_factor=scale_factor, drop_rate=self.drop_rate)
        else:
            embed_dims = [self.model.embed_dim * 2 ** i for i in self.extract_layers]
            self.feature_upsampler = SwinViTFeatureUpsampler(
                embed_dims, drop_rate=self.drop_rate
            )
        self.out_channels = self.feature_upsampler.out_channels

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate feature maps from the model at specified layers."""
        features = self.model.forward_intermediates(
            x,
            indices=self.extract_layers,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True
            )
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass that extracts features and upsamples them."""
        features = self.forward_features(x)
        features_upscaled = self.feature_upsampler(x, features)
        return features_upscaled


class Decoder(nn.Module):
    """
    Decoder for the UNETR / CellViT model that is an usual U-Net Decoder.

    Args:
        encoder_out_channels (list): List of output channels from the encoder.
        out_channels (int): Number of output channels for the decoder. Defaults to 32.
        drop_rate (float): Dropout rate for the decoder. Defaults to 0.
    """

    def __init__(
        self,
        encoder_out_channels: List[int],
        out_channels: int = 32,
        drop_rate: float = 0,
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()

        if len(encoder_out_channels) != 5:
            raise ValueError(f"Encoder should return 5 features, got {len(encoder_out_channels)}")
        embed_dim = encoder_out_channels[-1]
        bottleneck_dim = encoder_out_channels[3]
        decoder2_dim = encoder_out_channels[2]
        decoder3_dim = encoder_out_channels[1]
        decoder4_dim = encoder_out_channels[0]
        self.drop_rate = drop_rate

        self.bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                bottleneck_dim * 2, bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                bottleneck_dim, bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                bottleneck_dim, bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=bottleneck_dim,
                out_channels=decoder2_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder2_upsampler = nn.Sequential(
            Conv2DBlock(decoder2_dim * 2, decoder2_dim, dropout=self.drop_rate),
            Conv2DBlock(decoder2_dim, decoder2_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=decoder2_dim,
                out_channels=decoder3_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder1_upsampler = nn.Sequential(
            Conv2DBlock(decoder3_dim * 2, decoder3_dim, dropout=self.drop_rate),
            Conv2DBlock(decoder3_dim, decoder3_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=decoder3_dim,
                out_channels=decoder4_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder0_header = nn.Sequential(
            Conv2DBlock(decoder4_dim * 2, decoder4_dim, dropout=self.drop_rate),
            Conv2DBlock(decoder4_dim, decoder4_dim, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=decoder4_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        initialize_decoder_head(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet decoder from skip connection features."""
        z0, z1, z2, z3, z4 = features

        b4 = self.bottleneck_upsampler(z4)
        b3 = self.decoder3_upsampler(torch.cat([z3, b4], dim=1))
        b2 = self.decoder2_upsampler(torch.cat([z2, b3], dim=1))
        b1 = self.decoder1_upsampler(torch.cat([z1, b2], dim=1))
        decoder_output = self.decoder0_header(torch.cat([z0, b1], dim=1))
        return decoder_output


class UnetR(nn.Module):
    """
    UNETR/CellViT Model, a U-Net like network with vision transformer as backbone encoder.

    Skip connections are shared between branches, but each network has a distinct encoder.

    Args:
        img_size (int): Input image size.
        encoder_name (str): Name of the encoder backbone.
        encoder_weights (str, optional): Path to encoder weights. Defaults to None.
        decoder_out_channels (int, optional): Number of output channels for decoder. Defaults to 32.
        head_use_attention (bool, optional): Whether to use attention in segmentation head.
            Defaults to True.
        use_lora (bool, optional): Whether to use LoRA adaptation. Defaults to False.
        classes (int, optional): Number of output classes. Defaults to 1.
        activation (callable, optional): Activation function for segmentation head.
            Defaults to nn.Tanh().
        drop_rate (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(
            self,
            img_size: int,
            encoder_name: str,
            encoder_weights: str = None,
            decoder_out_channels: int = 32,
            head_use_attention: bool = True,
            use_lora: bool = False,
            classes: int = 1,
            activation=nn.Tanh(),
            drop_rate: float = 0,
    ) -> None:
        super().__init__()
        if encoder_name == "restnet50_lunit_swav":
            encoder = Resnet50LunitSwav(ckpt_path=encoder_weights,
                                        drop_rate=drop_rate)
        elif encoder_name in FOUNDATION_MODEL_REGISTRY.keys():
            encoder = ViTPyramidEncoder(img_size, encoder_name,
                                        ckpt_path=encoder_weights,
                                        drop_path_rate=drop_rate,
                                        use_lora=use_lora)
        else:
            try:
                encoder = smp.encoders.get_encoder(encoder_name, in_channels=3,
                                                   depth=4, weights="imagenet")
            except KeyError:
                raise ValueError(f"Unkown encoder, got {encoder_name}")

        self.encoder = encoder
        self.decoder = Decoder(self.encoder.out_channels,
                               out_channels=decoder_out_channels,
                               drop_rate=drop_rate)
        self.num_heads = classes
        for idx in range(self.num_heads):
            setattr(self, f'segmentation_head_{idx}', SegmentationHead(
                in_channels=decoder_out_channels,
                out_channels=1,
                activation=activation,
                kernel_size=3,
                use_attention=head_use_attention
            ))
        self.initialize()

    def initialize(self) -> None:
        """Initialize decoder, encoder feature upsampler (if applicable) and segmentation heads."""
        initialize_decoder_head(self.decoder)
        if isinstance(self.encoder, ViTPyramidEncoder):
            initialize_decoder_head(self.encoder.feature_upsampler)
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            initialize_decoder_head(segmentation_head)

    def freeze_encoder(self) -> None:
        """Freeze encoder to not train it."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        if hasattr(self.encoder, "feature_upsampler"):
            for p in self.encoder.feature_upsampler.parameters():
                p.requires_grad = True

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder to train the whole model."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of UNETR on input image x."""
        features = self.encoder(x)
        output_decoder = self.decoder(features)
        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(output_decoder)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs
