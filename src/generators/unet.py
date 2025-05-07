import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models import VisionTransformer, SwinTransformer
import segmentation_models_pytorch as smp

from .foundation_models import FOUNDATION_MODEL_REGISTRY
from .lora import apply_lora


class Unet(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with
    vision transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        encoder (int): Vit encoder of Unet
        num_classes (int): Number of output classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
    """

    def __init__(
        self,
        img_size,
        encoder_name,
        encoder_weights=None,
        decoder_out_channels=32,
        head_use_attention=True,
        use_lora=False,
        classes=1,
        activation=nn.Tanh(),
        drop_rate: float = 0,
        ):
        super().__init__()
        if encoder_name == "restnet50_lunit_swav":
            encoder = Resnet50LunitSwav(ckpt_path=encoder_weights, drop_rate=drop_rate)
        elif encoder_name in FOUNDATION_MODEL_REGISTRY.keys():
            encoder = ViTPyramidEncoder(img_size, encoder_name, ckpt_path=encoder_weights, drop_path_rate=drop_rate, use_lora=use_lora)
        else:
            try:
                encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=4, weights="imagenet")
            except KeyError:
                raise ValueError(f"Unkown encoder, got {encoder_name}")

        self.encoder = encoder
        self.decoder = Decoder(self.encoder.out_channels, out_channels=decoder_out_channels, drop_rate=drop_rate)
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

    def initialize(self):
        initialize_decoder_head(self.decoder)
        if isinstance(self.encoder, ViTPyramidEncoder):
            initialize_decoder_head(self.encoder.feature_upsampler)
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            initialize_decoder_head(segmentation_head)

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            p.requires_grad = False
        if hasattr(self.encoder, "feature_upsampler"):
            for layer_name, p in self.encoder.feature_upsampler.named_parameters():
                p.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True


    def forward(self, x):
        features = self.encoder(x)
        output_decoder = self.decoder(features)
        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(output_decoder)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Resnet50LunitSwav(nn.Module):
    def __init__(self, ckpt_path=None, drop_rate=0.):
        super().__init__()
        self.model = FOUNDATION_MODEL_REGISTRY["restnet50_lunit_swav"](
                ckpt_path=ckpt_path, drop_rate=drop_rate)
        self.convsteam = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=drop_rate),
            Conv2DBlock(32, 64, 3, dropout=drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.out_channels = [64, 64, 256, 512, 1024]
    
    def forward(self, x):
        features_convsteam = self.convsteam(x)
        features = self.model.forward_intermediates(
            x,
            indices=(0, 1, 2, 3),
            intermediates_only=True
            )
        return [features_convsteam] + features


class ViTPyramidEncoder(nn.Module):

    def __init__(self, img_size, encoder_name, ckpt_path=None, drop_path_rate=0., use_lora=False):
        super().__init__()
        try:
            model = FOUNDATION_MODEL_REGISTRY[encoder_name](
                img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate)
        except KeyError:
            raise NotImplementedError(f"Unknown model: try ones in {list(FOUNDATION_MODEL_REGISTRY.keys())}")
        if not isinstance(model, (VisionTransformer, SwinTransformer)):
            raise ValueError(f"Model should be a VisionTransformer or SwinTransformer, got {type(model)}")
        self.model = model
        if use_lora:
            apply_lora(self.model, rank=8, alpha=1.)
        
        is_vit = isinstance(self.model, VisionTransformer)
        depth = len(model.blocks) if is_vit else len(model.layers)
        if depth == 4:
            self.extract_layers = [0, 1, 2, 3]
        elif depth > 4:
            self.extract_layers = np.round(np.linspace(depth // 4, depth - 1, 4)).astype(int).tolist()
        else:
            raise ValueError("Vit Should have a depth higher than 3")

        self.patch_size = 16
        self.drop_rate = drop_path_rate

        assert img_size % self.patch_size == 0

        if is_vit:
            real_patch_size = self.model.patch_embed.patch_size[0]
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


    def forward_features(self, x):
        features = self.model.forward_intermediates(x,
                indices=self.extract_layers,
                norm=False, ####
                output_fmt="NCHW",
                intermediates_only=True
                )
        return features

    def forward(self, x):
        features = self.forward_features(x)
        features_upscaled = self.feature_upsampler(x, features)
        return features_upscaled


class ViTFeatureUpsampler(nn.Module):
    def __init__(self, embed_dim, drop_rate, scale_factor=None):
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
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest") if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.upsampler1 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest") if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.upsampler2 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest") if self.scale_factor else nn.Identity(),
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3
        self.upsampler3 = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode="nearest") if self.scale_factor else nn.Identity(),
            
        )  # skip connection 3
        self.out_channels = [
            self.convsteam[-1].out_channels,
            self.upsampler0[-1].out_channels,
            self.upsampler1[-1].out_channels,
            self.upsampler2[-1].out_channels,
            self.embed_dim,
        ]
        initialize_decoder_head(self)


    def forward(self, x, features):
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
    def __init__(self, embed_dims, drop_rate):
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


    def forward(self, x, features):
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


class Decoder(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with
    vision transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        encoder (int): Vit encoder of Unet
        num_classes (int): Number of output classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
    """

    def __init__(
        self,
        encoder_out_channels,
        out_channels=32,
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

    def forward(self, features: torch.Tensor) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well.
            Defaults to False.

        Returns:
            output: output of the Unet
        """

        z0, z1, z2, z3, z4 = features

        b4 = self.bottleneck_upsampler(z4)
        b3 = self.decoder3_upsampler(torch.cat([z3, b4], dim=1))
        b2 = self.decoder2_upsampler(torch.cat([z2, b3], dim=1))
        b1 = self.decoder1_upsampler(torch.cat([z1, b2], dim=1))
        decoder_output = self.decoder0_header(torch.cat([z0, b1], dim=1))

        return decoder_output


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
        activation = activation
        super().__init__(attention, conv2d, activation)
        initialize_decoder_head(self)


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

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

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d,
    batch-normalisation, ReLU activation and dropout

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

    def forward(self, x):
        return self.block(x)


def initialize_decoder_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
