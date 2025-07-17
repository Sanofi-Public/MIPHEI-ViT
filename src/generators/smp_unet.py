"""
Implementation of U-Net and its variants using the segmentation_models_pytorch library.

Inspired: https://github.com/qubvel-org/segmentation_models.pytorch/tree/main/segmentation_models_pytorch/decoders/unet
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch import Unet as UnetSMP
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder


def initialize_decoder_head(module: nn.Module) -> None:
    """
    Initialize the weights and biases of decoder head layers following Pix2pix initialization.

    Apply Pix2Pix normal initialization to Conv2d and ConvTranspose2d weights, set their biases to
    zero, and initialize BatchNorm2d weights and biases.

    Args:
        module (nn.Module): The decoder head module whose layers will be initialized.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class AttentionBlock(nn.Module):
    """
    Attention block for focusing on important regions.

    This block generates an attention map using a small convolutional network and applies it to the
    input feature map, allowing the network to focus on important regions.
    Inspired by https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py#L108
        from paper Attention U-Net: Learning Where to Look for the Pancreas
        https://arxiv.org/abs/1804.03999

    Args:
        in_chns (int): Number of input channels.

    Attributes:
        psi (nn.Sequential): Sequential module that generates the attention map.

    """

    def __init__(self, in_chns: int):
        super(AttentionBlock, self).__init__()
        # Attention generation
        self.psi = nn.Sequential(
            nn.Conv2d(in_chns, in_chns // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_chns // 2),
            nn.ReLU(),
            nn.Conv2d(in_chns // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that applies attention gating to the input tensor x."""
        # Project decoder output to intermediate space
        g = self.psi(x)
        return x * g


class SegmentationHead(nn.Sequential):
    """
    Final U-Net head module for dense image task (segmentation, image translation).

    This module applies an optional attention mechanism, followed by a convolutional layer and
    an activation function. It is typically used as the final layer in segmentation models
    to produce the output segmentation map.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (number of markers).
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        activation (callable or None, optional): Activation function to apply after convolution.
            If None, no activation is applied.
        use_attention (bool, optional): If True, applies an attention block before the convolution.
            Defaults to False.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: bool = None,
        use_attention: bool = False,
    ):
        if use_attention:
            attention = AttentionBlock(in_channels)
        else:
            attention = nn.Identity()
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        activation = activation  # Activation(activation)
        super().__init__(attention, conv2d, activation)
        initialize_decoder_head(self)


class InterpDecoderBlock(nn.Module):
    """
    Decoder upsampling block with interpolation, skip connection, and optional attention.

    This block performs upsampling using nearest neighbor interpolation, concatenates skip
    connections, applies attention mechanisms, and processes the result through two convolutional
    layers with optional batch normalization.
    Args:
        in_channels (int): Number of input channels from the previous layer.
        skip_channels (int): Number of channels from the skip connection.
        out_channels (int): Number of output channels after processing.
        use_batchnorm (bool, optional): If True, applies batch normalization after convolutions.
            Defaults to True.
        attention_type (str or None, optional): Type of attention mechanism to use.
            If None, attention is not applied.
    Attributes:
        conv1 (nn.Module): First convolutional block (Conv2dReLU).
        attention1 (nn.Module): Attention mechanism applied after concatenation (if specified).
        conv2 (nn.Module): Second convolutional block (Conv2dReLU).
        attention2 (nn.Module): Attention mechanism applied after the second convolution
            (if specified).
    Methods:
        forward(x, skip=None):
            Forward pass of the decoder block.
            Args:
                x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
                skip (torch.Tensor or None, optional): Skip connection tensor of shape
                    (N, skip_channels, H*2, W*2).
            Returns:
                torch.Tensor: Output tensor of shape (N, out_channels, H*2, W*2).
    Reference:
        https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py#L10
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for upsampling with optional skip connection and attention."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    """
    U-Net Decoder module.

    This class implements the decoder part of a U-Net architecture, which reconstructs
    the spatial resolution of the input by combining upsampled feature maps with skip
    connections from the encoder. It supports optional batch normalization, attention
    mechanisms, and a center block (only used if VGG as encoder).
    Args:
        encoder_channels (List[int]): Number of channels for each feature map from the encoder.
        decoder_channels (List[int]): Number of output channels for each decoder block.
        n_blocks (int, optional): Number of decoding blocks. Defaults to 5.
        use_batchnorm (bool, optional): If True, applies batch normalization in decoder blocks.
            Defaults to True.
        attention_type (str or None, optional): Type of attention mechanism to use in decoder
            blocks. Defaults to None.
        center (bool, optional): If True, adds a center block between encoder and decoder.
            Useful if VGG encoder. Defaults to False.
    Attributes:
        center (nn.Module): Center block or identity mapping.
        blocks (nn.ModuleList): List of decoder blocks.
    Raises:
        ValueError: If the number of decoder channels does not match the number of blocks.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type=None,
        center: bool = False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = {"use_batchnorm": use_batchnorm, "attention_type": attention_type}
        blocks = [
            InterpDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder."""
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class UnetMultiHeads(SegmentationModel):
    """
    U-Net model with multiple final heads for multi-class/channel dense prediction.

    This model extends a standard U-Net architecture by allowing multiple independent final heads,
    each producing a separate output channel. Useful for scenarios where each class or task
    requires a dedicated output head.
    Args:
        encoder_name (str): Name of the encoder backbone (default: "resnet34").
        encoder_depth (int): Number of stages in the encoder (default: 5).
        encoder_weights (str): Pretrained weights for the encoder (default: "imagenet").
        decoder_use_batchnorm (bool): If True, use BatchNorm in decoder (default: True).
        decoder_channels (tuple of int): Number of channels for each decoder block
            (default: (256, 128, 64, 32, 16)).
        decoder_attention_type (str): Type of attention mechanism in decoder (default: None).
        in_channels (int): Number of input channels (default: 3).
        classes (int): Number of heads for each class / channel (default: 1).
        activation (callable or None): Activation function for final heads (default: None).
        dropout (float or None): Dropout probability for decoder blocks (default: None).
        use_attention (bool): If True, use attention in final heads (default: True).
    Attributes:
        encoder (nn.Module): Encoder backbone.
        decoder (nn.Module): U-Net decoder.
        num_heads (int): Number of final heads.
        segmentation_head_{idx} (SegmentationHead): Segmentation head for each final head.
        classification_head (None): Placeholder for optional classification head.
        name (str): Model name.
    Methods:
        initialize():
            Initializes decoder and final heads.
        forward(x):
            Forward pass through the model.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: int = (256, 128, 64, 32, 16),
        decoder_attention_type: str = None,
        in_channels: int = 3,
        classes: int = 1,
        activation=None,
        dropout: float = None,
        use_attention: bool = True
    ):
        super().__init__()

        self.encoder = smp_get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.num_heads = classes
        # Create decoders and segmentation heads as attributes
        self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
        for idx in range(self.num_heads):
            setattr(self, f'segmentation_head_{idx}', SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=1,
                activation=Activation(activation),
                kernel_size=3,
                use_attention=use_attention
            ))

        self.classification_head = None
        if dropout:
            for idx in range(1, 3):
                self.decoder.blocks[idx].conv1.add_module(
                    '3', nn.Dropout2d(p=dropout))
        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self) -> None:
        """Initialize decoder, heads, and classification head if present."""
        initialize_decoder_head(self.decoder)
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            initialize_decoder_head(segmentation_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder, decoder, and heads."""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(decoder_output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)

        return outputs


class UnetMultiHeadsFG(UnetMultiHeads):
    """
    U-Net model with multiple final heads and an additional foreground segmentation head.

    This model extends UnetMultiHeads by adding a dedicated foreground segmentation head,
    allowing the model to output both multi-head predictions and a separate foreground mask.
    This dual-head design enables the model to be guided by both dense prediction and explicit
    foreground segmentation objectives, which can lead to more effective learning in tasks where
    foreground information helps guide the main prediction process (such as H&E to mIF translation).
    Args:
        encoder_name (str): Name of the encoder backbone (default: "resnet34").
        encoder_depth (int): Number of stages in the encoder (default: 5).
        encoder_weights (str): Pretrained weights for the encoder (default: "imagenet").
        decoder_use_batchnorm (bool): If True, use BatchNorm in decoder (default: True).
        decoder_channels (tuple of int): Number of channels for each decoder block
            (default: (256, 128, 64, 32, 16)).
        decoder_attention_type (str): Type of attention mechanism in decoder (default: None).
        in_channels (int): Number of input channels (default: 3).
        classes (int): Number of heads for each class / channel (default: 1).
        activation (callable or None): Activation function for final heads (default: None).
        dropout (float or None): Dropout probability for decoder blocks (default: None).
    Attributes:
        encoder (nn.Module): Encoder backbone.
        decoder (nn.Module): U-Net decoder.
        num_heads (int): Number of final heads.
        segmentation_head_{idx} (SegmentationHead): Segmentation head for each final head.
        foreground_head (SegmentationHead): Segmentation head for foreground prediction.
        classification_head (None): Placeholder for optional classification head.
        name (str): Model name.
    Methods:
        initialize():
            Initializes decoder, heads, and foreground head.
        forward(x):
            Forward pass through encoder, decoder, heads, and foreground head.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Concatenated outputs from all final heads (B, num_heads, H, W).
            - Foreground segmentation output (B, classes, H, W).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: int = (256, 128, 64, 32, 16),
        decoder_attention_type: str = None,
        in_channels: int = 3,
        classes: int = 1,
        activation=None,
        dropout: float = None,
    ):
        super().__init__(
            encoder_name=encoder_name, encoder_depth=encoder_depth,
            encoder_weights=encoder_weights, decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels, decoder_attention_type=decoder_attention_type,
            in_channels=in_channels, classes=classes, activation=activation, dropout=dropout
        )

        self.foreground_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=Activation(activation),
                kernel_size=3)

        initialize_decoder_head(self.foreground_head)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass: returns segmentation and foreground masks."""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        output_fg = self.foreground_head(decoder_output)

        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(decoder_output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)

        return outputs, output_fg


# =============================================================================
# The following classes are not used
# =============================================================================


class Unet(UnetSMP):  # not used
    """
    U-Net extension with optional dropout and Pix2Pix initialization.

    This class extends the UnetSMP model, adding optional dropout layers to the decoder
    and using in-place ReLU activations as recommended for Pix2Pix (see:
    https://www.aryan.no/post/pix2pix/pix2pix/).
    It also provides a method to initialize decoder and head weights following Pix2Pix
    initialization.
    Args:
        dropout (float, optional): Dropout probability to apply to the decoder blocks 1 and 2.
        *args: Additional positional arguments passed to the UnetSMP base class.
        **kwargs: Additional keyword arguments passed to the UnetSMP base class.
    Attributes:
        Same as UnetSMP.
    Methods:
        Same as UnetSMP, with the following addition:
        initialize():
            Initializes the decoder, head, and classification head (if present)
            using the `initialize_decoder_head` function.
    """

    def __init__(self, dropout: float = None, *args, **kwargs):
        # https://www.aryan.no/post/pix2pix/pix2pix/
        super().__init__(*args, **kwargs)
        if dropout:
            for idx in range(1, 3):
                self.decoder.blocks[idx].conv1.add_module(
                    '3', nn.Dropout2d(p=dropout))
        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def initialize(self) -> None:
        """Initialize decoder and heads with Pix2pix initialization."""
        initialize_decoder_head(self.decoder)
        initialize_decoder_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)


class UnetTwoHeads(SegmentationModel):  # not used
    """
    U-Net model with two decoders: image translation and foreground segmentation.

    This model extends the standard U-Net by using two decoders for H&E to mIF translation.
    The first decoder is intended for image translation tasks, while the second decoder focuses on
    foreground segmentation. This dual-head design allows the model to be guided by both
    translation and segmentation objectives, enabling more effective learning in tasks where
    foreground information can help guide the translation process (like H&E to mIF).
    Args:
        encoder_name (str): Name of the encoder backbone to use (default: "resnet34").
        encoder_depth (int): Number of stages in the encoder (default: 5).
        encoder_weights (str): Pretrained weights for the encoder (default: "imagenet").
        decoder_use_batchnorm (bool): Whether to use BatchNorm in the decoder (default: True).
        decoder_channels (tuple of int): Number of channels for each decoder block
            (default: (256, 128, 64, 32, 16)).
        decoder_attention_type (str, optional): Type of attention mechanism to use in the decoder
            (default: None).
        in_channels (int): Number of input channels (default: 3).
        classes (int): Number of output classes for segmentation (default: 1).
        activation (callable, optional): Activation function to apply to the output (default: None).
        dropout (float, optional): Dropout probability to apply in decoder blocks (default: None).
    Attributes:
        encoder (nn.Module): Encoder backbone.
        decoder (nn.Module): Decoder for image translation.
        foregound_decoder (nn.Module): Decoder for foreground segmentation.
        segmentation_head (nn.Module): Head for image translation output.
        foreground_head (nn.Module): Head for foreground segmentation output.
        classification_head (None): Placeholder for classification head (not used).
        name (str): Model name.
    Methods:
        initialize():
            Initializes decoder and head weights.
        forward(x):
            Forward pass through the model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predicted translated image (or segmentation) and
            foreground segmentation mask.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: int = (256, 128, 64, 32, 16),
        decoder_attention_type: str = None,
        in_channels: int = 3,
        classes: int = 1,
        activation=None,
        dropout: float = None,
    ):
        super().__init__()

        self.encoder = smp_get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.foregound_decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=Activation(activation),
            kernel_size=3,
        )

        self.foreground_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=Activation(None),
            kernel_size=3,
        )

        self.classification_head = None
        if dropout:
            for idx in range(1, 3):
                self.decoder.blocks[idx].conv1.add_module(
                    '3', nn.Dropout2d(p=dropout))
                self.foreground_head.blocks[idx].conv1.add_module(
                    '3', nn.Dropout2d(p=dropout))
        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self) -> None:
        """Initialize all decoder and head modules of the model."""
        initialize_decoder_head(self.decoder)
        initialize_decoder_head(self.foregound_decoder)
        initialize_decoder_head(self.segmentation_head)
        initialize_decoder_head(self.foreground_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass and returns segmentation masks and foreground mask."""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        foreground_decoder_output = self.foregound_decoder(*features)

        masks = self.segmentation_head(decoder_output)
        mask_foreground = self.foreground_head(foreground_decoder_output)

        return masks, mask_foreground
