import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch import Unet as UnetSMP
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock
from segmentation_models_pytorch.base.modules import Activation


class InterpDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
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

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")# mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        # Replace interpolation with ConvTranspose2d
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upconv(x)  # Upsample using ConvTranspose2d
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
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
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            InterpDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class Unet(UnetSMP):
    def __init__(self, dropout=None, *args, **kwargs):
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

    def initialize(self):
        initialize_decoder_head(self.decoder)
        initialize_decoder_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)


class UnetTwoHeads(SegmentationModel):

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
        activation = None,
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
            activation=activation,
            kernel_size=3,
        )

        self.foreground_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
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

    def initialize(self):
        initialize_decoder_head(self.decoder)
        initialize_decoder_head(self.foregound_decoder)
        initialize_decoder_head(self.segmentation_head)
        initialize_decoder_head(self.foreground_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        foreground_decoder_output = self.foregound_decoder(*features)

        masks = self.segmentation_head(decoder_output)
        mask_foreground = self.foreground_head(foreground_decoder_output)

        return masks, mask_foreground


class UnetMultiHeads(SegmentationModel):

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
        activation = None,
        dropout: float = None,
        use_attention=True
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
                activation=activation,
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

    def initialize(self):
        initialize_decoder_head(self.decoder)
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            initialize_decoder_head(segmentation_head)
        if self.classification_head is not None:
            initialize_decoder_head(self.classification_head)

    def forward(self, x):
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
        activation = None,
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
                activation=activation,
                kernel_size=3)

        initialize_decoder_head(self.foreground_head)


    def forward(self, x):
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


class AttentionBlock(nn.Module):
    def __init__(self, in_chns):
        super(AttentionBlock, self).__init__()
        # Attention generation
        self.psi = nn.Sequential(
            nn.Conv2d(in_chns, in_chns // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_chns // 2),
            nn.ReLU(),
            nn.Conv2d(in_chns // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project decoder output to intermediate space
        g = self.psi(x)
        return x * g


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, use_attention=False,
    ):
        if use_attention:
            attention = AttentionBlock(in_channels)
        else:
            attention = nn.Identity()
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        activation = Activation(activation)
        super().__init__(attention, conv2d, activation)


def initialize_decoder_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
