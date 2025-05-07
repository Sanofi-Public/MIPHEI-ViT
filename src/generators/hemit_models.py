import torch
import torch.nn as nn
from torch.nn import init
import functools
import timm
import math
from torch import einsum
from einops import rearrange


###############################################################################
# Helper Functions
###############################################################################


def get_generator_hemit(input_nc, output_nc, image_size, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_1024':
        net = UnetGenerator(input_nc, output_nc, 10, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'swinT':
        config = {
            "model_params": {
                "img_size": image_size,
                "output_nc": output_nc,
                "patch_size": 4,
                "window_size": 8,
                "in_chans": 3,
                "depths": [1, 1, 3, 1],
                "embed_dim": 96,
                "cnn_channels": [16, 32, 64],
            }
        }
        net = HybridSwinT(**config["model_params"])
    elif netG == 'swinT_old':
        config = {
            "model_params": {
                "img_size": image_size,
                "output_nc": output_nc,
                "patch_size": 4,
                "window_size": 8,
                "in_chans": 3,
                "depths": [2, 2, 6, 2],
                "embed_dim": 96,
            }
        }
        net = HybridSwinT_2(**config["model_params"])
    elif netG == 'SwinTUnet':
        config = {
            "model_params": {
                "img_size": image_size,
                "output_nc": output_nc,
                "patch_size": 32,
                "window_size": 64,
                "depths": [2, 2, 6, 2],
                "embed_dim": 96
            }
        }
        net = SwinUnetGenerator(**config["model_params"])
    elif netG == 'SwinTResnet':
        config = {
            "model_params": {
                "input_nc": input_nc,
                "output_nc": output_nc,
                "img_size": image_size,
                "patch_size": 32,
                "window_size": 64,
                "depths": [2, 2, 6, 2],
                "embed_dim": 96
            }
        }
        net = ResnetGeneratorSwinT(**config["model_params"])
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer in [nn.InstanceNorm2d, nn.Identity]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class HybridSwinT(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=3, cnn_channels=[16, 32, 64], **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))
        self.cnn_depths = cnn_channels

        # Customizable depths for the CNN module
        layers = []
        in_channels = in_chans
        for channel in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half
            ])
            in_channels = channel
        self.cnn_block = nn.Sequential(*layers)

        # create the Swin Transformer model
        self.model = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=64,  # The input channels for the SwinTransformer is now 64
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.model.head = nn.Identity()  # remove classification head

        self.decoder = Decoder_hbrid(in_channs=int(embed_dim * (math.pow(2, len(depths) - 1))),
                               output_channels=output_channels,
                               patch_size=patch_size,
                               )

    def forward(self, x):
        self.stage_outputs = []
        # Pass the input through the cnn_block first
        # x = self.cnn_block(x)
        for layer in self.cnn_block:
            x = layer(x)
            print(x.shape)
            if isinstance(layer, nn.MaxPool2d):
                self.stage_outputs.append(x)

        B, C, H, W = x.shape

        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for stage in self.model.layers:
            for blk in stage.blocks:
                x = blk(x)
                print(x.shape)
            if stage.downsample is not None:
                self.stage_outputs.append(x)
                x = stage.downsample(x)

        x = self.model.norm(x)
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths) - 1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths) - 1)), self.last_stage_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x, self.stage_outputs)

        return x


class Decoder_hbrid(nn.Module):
    def __init__(self, in_channs, output_channels, patch_size):
        super().__init__()

        '''
        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)
        '''

        # Replace ConvTranspose with Upsample + Convolution
        self.upsample_and_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)
        )
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample_and_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs // 2, in_channs // 4, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs // 4, kernel_size=3, padding=1)

        self.upsample_and_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        #upsampling_factor = int(math.log(patch_size, 2)) + 3
        '''
        self.final_upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(in_channs // 8, output_channels if i == upsampling_factor - 1 else in_channs // 8,
                                kernel_size=2, stride=2) for i
             in range(upsampling_factor)])
        '''


        self.upsample_and_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1)
        )
        self.upsample_and_conv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1)
        )

        self.conv4 = nn.Conv2d(88, 44, kernel_size=3, padding=1)

        self.upsample_and_conv6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(44, 22, kernel_size=3, padding=1)
        )

        self.conv5 = nn.Conv2d(54, 27, kernel_size=3, padding=1)

        self.upsample_and_conv7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(27, 14, kernel_size=3, padding=1)
        )

        self.conv6 = nn.Conv2d(30, 15, kernel_size=3, padding=1)

        self.upsample_and_conv8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(15, 3, kernel_size=3, padding=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, stage_outputs):
        x = self.upsample_and_conv1(x)
        h_w_dim = int((stage_outputs[-1].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-1].view(stage_outputs[-1].shape[0], h_w_dim, h_w_dim, stage_outputs[-1].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv1(x)

        x = self.upsample_and_conv2(x)
        h_w_dim2 = int((stage_outputs[-2].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-2].view(stage_outputs[-2].shape[0], h_w_dim2, h_w_dim2, stage_outputs[-2].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv2(x)

        x = self.upsample_and_conv3(x)
        h_w_dim3 = int((stage_outputs[-3].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-3].view(stage_outputs[-3].shape[0], h_w_dim3, h_w_dim3, stage_outputs[-3].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv3(x)

        x = self.upsample_and_conv4(x)
        x = self.upsample_and_conv5(x)

        x = torch.cat((x, stage_outputs[-4]), dim=1)
        x = self.conv4(x)

        x = self.upsample_and_conv6(x)

        x = torch.cat((x, stage_outputs[-5]), dim=1)
        x = self.conv5(x)

        x = self.upsample_and_conv7(x)

        x = torch.cat((x, stage_outputs[-6]), dim=1)
        x = self.conv6(x)

        x = self.upsample_and_conv8(x)


        return self.tanh(x)


class HybridSwinT_2(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=3, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))

        # Add a block of convolutional layers
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
        )

        # create the Swin Transformer model
        self.model = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=64,  # The input channels for the SwinTransformer is now 64
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.model.head = nn.Identity()  # remove classification head

        self.decoder = Decoder_hbrid_2(in_channs=int(embed_dim * (math.pow(2, len(depths) - 1))),
                               output_channels=output_channels,
                               patch_size=patch_size,
                               )

    def forward(self, x):
        self.stage_outputs = []
        # Pass the input through the cnn_block first
        x = self.cnn_block(x)
        B, C, H, W = x.shape

        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for stage in self.model.layers:
            for blk in stage.blocks:
                x = blk(x)
            if stage.downsample is not None:
                self.stage_outputs.append(x)
                x = stage.downsample(x)

        x = self.model.norm(x)
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths) - 1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths) - 1)), self.last_stage_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x, self.stage_outputs)

        return x


class Decoder_hbrid_2(nn.Module):
    def __init__(self, in_channs, output_channels, patch_size):
        super().__init__()

        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        upsampling_factor = int(math.log(patch_size, 2))+2
        self.final_upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(in_channs // 8, output_channels if i == upsampling_factor - 1 else in_channs // 8,
                                kernel_size=2, stride=2) for i
             in range(upsampling_factor)])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, stage_outputs):
        x = self.upsample1(x)
        h_w_dim = int((stage_outputs[-1].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-1].view(stage_outputs[-1].shape[0], h_w_dim, h_w_dim, stage_outputs[-1].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv1(x)

        x = self.upsample2(x)
        h_w_dim2 = int((stage_outputs[-2].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-2].view(stage_outputs[-2].shape[0], h_w_dim2, h_w_dim2, stage_outputs[-2].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv2(x)

        x = self.upsample3(x)
        h_w_dim3 = int((stage_outputs[-3].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-3].view(stage_outputs[-3].shape[0], h_w_dim3, h_w_dim3, stage_outputs[-3].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv3(x)

        for upsample_layer in self.final_upsample_layers:
            x = upsample_layer(x)

        return self.tanh(x)


class SwinUnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=10, ngf=64, norm_layer=nn.BatchNorm2d,
                 norm_layer_swinT=nn.LayerNorm, use_dropout=False,
                 img_size=1024, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2):
        super(SwinUnetGenerator, self).__init__()

        # UNet branch
        # Downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(num_downs):
            in_channels = input_nc if i == 0 else min(ngf * (2 ** (i - 1)), ngf * 8)
            out_channels = min(ngf * (2 ** i), ngf * 8)
            is_innermost = (i == num_downs - 1)  # Check if it's the innermost layer
            self.down_layers.append(
                self.get_down_layer(in_channels, out_channels, norm_layer, use_dropout, is_innermost))


        # Upsampling layers
        self.up_layers = nn.ModuleList()
        # Define channel configurations for each layer

        channel_configs = [
            (ngf * 8, ngf * 8),  # i == 0 (innermost)
            (ngf * 16, ngf * 8),  # i == 1
            (ngf * 16, ngf * 8),  # i == 2
            (ngf * 16, ngf * 8),  # i == 3
            (ngf * 16, ngf * 8),  # i == 4
            (ngf * 16, ngf * 8),  # i == 5
            (ngf * 16, ngf * 4),  # i == 6
            (ngf * 8, ngf * 2),  # i == 7
            (ngf * 4, ngf),  # i == 8
        ]

        '''
        channel_configs = [
            (ngf * 8, ngf * 8),  # i == 0 (innermost)
            (1024, 512),  # i == 1
            (1024, 512),  # i == 2
            (1024, 512),  # i == 3
            (1024, 512),  # i == 4
            (1024, 512),  # i == 5
            (1024, 512),  # i == 6
            (ngf * 12, ngf * 4),  # i == 7
            (ngf * 6, ngf * 2),  # i == 8
        ]
        '''
        for i in range(num_downs - 1):
            in_channels, out_channels = channel_configs[i]
            self.up_layers.append(self.get_up_layer(in_channels, out_channels, norm_layer, use_dropout))

        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 3, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Swin Transformer branch
        self.swinT = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer_swinT
        )
        self.swinT.head = nn.Identity()  # remove classification head

        self.cross_atts = nn.ModuleList([
            Cross_Att(ngf * 8, embed_dim * 2),
            Cross_Att(ngf * 8, embed_dim * 4),
            Cross_Att(ngf * 8, embed_dim * 8),
        ])

        self.cnn_starting_index = math.log2(patch_size)

    def get_down_layer(self, in_channels, out_channels, norm_layer, use_dropout, is_innermost):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        if not is_innermost:  # Only add normalization layer if it's not the innermost layer
            layers.append(norm_layer(out_channels))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def get_up_layer(self, in_channels, out_channels, norm_layer, use_dropout):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            norm_layer(out_channels)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x
        unet_features = []
        swinT_features = []
        skip_connections = []

        # unet processing
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
            if x.shape[2] == 16 or x.shape[2] == 8 or x.shape[2] == 4:
                unet_features.append(x)
                skip_connections[-1] = unet_features[-1]
        skip_connections = skip_connections[:-1] # Remove the last

        # SwinT processing
        B, C, H, W = x0.shape
        print('img shape: ', B, C, H, W)
        x2 = self.swinT.patch_embed(x0)
        x2 = self.swinT.pos_drop(x2)
        for stage in self.swinT.layers:
            for blk in stage.blocks:
                x2 = blk(x2)
            if stage.downsample is not None:
                x2 = stage.downsample(x2)
                h_w_dim = int((x2.shape[1]) ** 0.5)
                print('h_w_dim:', h_w_dim)
                swinT_features.append(x2.view(x2.shape[0], h_w_dim, h_w_dim,
                                              x2.shape[2]).permute(0, 3, 1, 2))

        # Apply Cross Attention at each scale
        for i in range(len(self.cross_atts)):
           # unet_features[i], swinT_features[i] = self.cross_atts[i](unet_features[i], swinT_features[i])
           unet_features[i] = self.cross_atts[i](unet_features[i], swinT_features[i])
        skip_connections[int(self.cnn_starting_index)] = unet_features[0]
        skip_connections[int(self.cnn_starting_index + 1)] = unet_features[1]
        skip_connections[int(self.cnn_starting_index + 2)] = unet_features[2]
        skip_connections = skip_connections[::-1]

        for up_layer, skip in zip(self.up_layers, skip_connections):
            x = up_layer(x)
            x = torch.cat([x, skip], dim=1)

        return self.final_layer(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Cross_Att(nn.Module):
    def __init__(self, dim_unet, dim_swinT):
        super().__init__()
        self.transformer_unet = Transformer(dim=dim_unet, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_swinT = Transformer(dim=dim_swinT, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_unet = nn.LayerNorm(dim_unet)
        self.norm_swinT = nn.LayerNorm(dim_swinT)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_unet = nn.Linear(dim_unet, dim_swinT)
        self.linear_swinT = nn.Linear(dim_swinT, dim_unet)
        self.gate = nn.Sequential(
            nn.Conv2d(dim_unet, dim_unet, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, unet_feature, swinT_feature):
        original_unet_feature = unet_feature.clone()
        gate_values = self.gate(original_unet_feature)
        b_u, c_u, h_u, w_u = unet_feature.shape
        unet_feature = unet_feature.reshape(b_u, c_u, -1).permute(0, 2, 1)
        b_s, c_s, h_s, w_s = swinT_feature.shape
        swinT_feature = swinT_feature.reshape(b_s, c_s, -1).permute(0, 2, 1)
        # unet_t = torch.flatten(self.avgpool(self.norm_swinT(unet_feature).transpose(1,2)), 1)
        swinT_t = torch.flatten(self.avgpool(self.norm_swinT(swinT_feature).transpose(1,2)), 1)
        # unet_t = self.linear_swinT(unet_t).unsqueeze(1)
        swinT_t = self.linear_swinT(swinT_t).unsqueeze(1)
        # swinT_feature = self.transformer_unet(torch.cat([unet_t, swinT_feature],dim=1))[:, 1:, :]
        unet_feature = self.transformer_unet(torch.cat([swinT_t, unet_feature],dim=1))[:, 1:, :]
        unet_feature = unet_feature.permute(0, 2, 1).reshape(b_u, c_u, h_u, w_u)
        # swinT_feature = swinT_feature.permute(0, 2, 1).reshape(b_s, c_s, h_s, w_s)
        # unet_feature_output = 0.2 * unet_feature + (1 - 0.2) * original_unet_feature
        unet_feature_output = gate_values * unet_feature + (1 - gate_values) * original_unet_feature

        # return unet_feature_output
        return unet_feature_output


class ResnetGeneratorSwinT(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',
                 norm_layer_swinT=nn.LayerNorm, img_size=1024, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2
                 ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorSwinT, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # Initial layers
        self.initial_layers = self.init_layers(input_nc, ngf, norm_layer, use_bias)

        # Downsampling layers
        self.downsampling_layers = self.downsample_layers(ngf, norm_layer, use_bias)

        # ResNet blocks
        self.resnet_blocks = self.resnet_blocks_layers(ngf, n_blocks, padding_type, norm_layer, use_dropout,
                                                       use_bias)

        # Swin Transformer branch
        self.swinT = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer_swinT
        )
        self.swinT.head = nn.Identity()  # remove classification head

        self.cross_atts = nn.ModuleList([
            GatedCrossAttention(cnn_channels=128, swinT_channels=192, upsample_factor=int(math.log2(patch_size))),
            GatedCrossAttention(cnn_channels=256, swinT_channels=384, upsample_factor=int(math.log2(patch_size))),
            GatedCrossAttention(cnn_channels=512, swinT_channels=768, upsample_factor=int(math.log2(patch_size)))
        ])

        self.patch_projector = nn.Conv2d(input_nc, embed_dim, kernel_size=1, stride=1, bias=True)

        # Upsampling layers
        self.upsampling_layers = self.upsample_layers(ngf, norm_layer, use_bias)

        # Final layers
        self.final_layers = self.final_layers_func(ngf, output_nc)

    def init_layers(self, input_nc, ngf, norm_layer, use_bias):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        return nn.Sequential(*layers)

    def downsample_layers(self, ngf, norm_layer, use_bias):
        n_downsampling = 3
        layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            block = nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            )
            layers.append(block)
        return layers

    def resnet_blocks_layers(self, ngf, n_blocks, padding_type, norm_layer, use_dropout, use_bias):
        mult = 2 ** 3  # 3 for n_downsampling
        layers = []
        for i in range(n_blocks):
            layers.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias))
        return nn.Sequential(*layers)

    def upsample_layers(self, ngf, norm_layer, use_bias):
        n_downsampling = 3
        layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            block = nn.Sequential(
                nn.ConvTranspose2d(ngf * mult * 2,  # Multiply by 2 to account for concatenation
                                   int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            )
            layers.append(block)
        return layers

    def final_layers_func(self, ngf, output_nc):
        return nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.initial_layers(input)

        # Downsample while storing the intermediate outputs
        downsampled_features = []
        for down_layer in self.downsampling_layers:
            x = down_layer(x)
            downsampled_features.append(x)

        x = self.resnet_blocks(x)

        # SwinT branch
        swinT_features = []
        # SwinT processing
        x2 = self.swinT.patch_embed(input)
        #x2 = self.swinT.pos_drop(x2)
        for i, stage in enumerate(self.swinT.layers):
            if stage.downsample is not None and i > 0:
                x2 = stage.downsample(x2)
                swinT_features.append(x2.permute(0, 3, 1, 2))
            for blk in stage.blocks:
                x2 = blk(x2)
            """if stage.downsample is not None:
                x2 = stage.downsample(x2)
                h_w_dim = int((x2.shape[1]) ** 0.5)
                swinT_features.append(x2.view(x2.shape[0], h_w_dim, h_w_dim,
                                              x2.shape[2]).permute(0, 3, 1, 2))"""

        # Apply Cross Attention at each scale
        for i in range(len(self.cross_atts)):
            downsampled_features[i] = self.cross_atts[i](downsampled_features[i], swinT_features[i])

        # Upsample with concatenation
        for up_layer, feature in zip(self.upsampling_layers, reversed(downsampled_features)):
            x = torch.cat([x, feature], dim=1)
            x = up_layer(x)

        x = self.final_layers(x)

        return x


class GatedCrossAttention(nn.Module):
    def __init__(self, cnn_channels, swinT_channels, num_heads=8, k=1000, upsample_factor=5):
        super(GatedCrossAttention, self).__init__()

        self.swinT_transform = nn.Conv2d(swinT_channels, cnn_channels, kernel_size=1)

        self.attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)

        self.gate = nn.Sequential(
            nn.Conv2d(cnn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        '''
        self.cnn_gate = nn.Sequential(
            nn.Conv2d(cnn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.swinT_gate = nn.Sequential(
            nn.Conv2d(cnn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        '''
        # Predefine the upsampling blocks based on the upsample_factor
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(cnn_channels, cnn_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
            ) for _ in range(upsample_factor)
        ])
        self.k = k

    def forward(self, downsampling_features, swinT_features):
        # Transform channel dimensions to common_channels
        swinT_features = self.swinT_transform(swinT_features)

        for upsample_block in self.upsample_blocks:
            swinT_features = upsample_block(swinT_features)

        # Calculate gate values
        gate_values = self.gate(downsampling_features)
        # cnn_gate_values = self.cnn_gate(downsampling_features)
        # swinT_gate_values = self.swinT_gate(swinT_features)

        # Flatten and permute for attention module
        down_features_flat = downsampling_features.flatten(2).permute(2, 0, 1)
        swinT_features_flat = swinT_features.flatten(2).permute(2, 0, 1)

        # Select top-k activations to apply attention
        _, top_indices = torch.topk(gate_values.view(gate_values.size(0), -1), k=self.k, dim=1)
        # _, cnn_top_indices = torch.topk(cnn_gate_values.view(cnn_gate_values.size(0), -1), k=self.k, dim=1)
        # _, swinT_top_indices = torch.topk(swinT_gate_values.view(swinT_gate_values.size(0), -1), k=self.k, dim=1)

        down_features_subset = torch.index_select(down_features_flat, 0, top_indices.view(-1))
        swinT_features_subset = torch.index_select(swinT_features_flat, 0, top_indices.view(-1))
        #cnn_features_subset = torch.index_select(down_features_flat, 0, cnn_top_indices.view(-1))
        #swinT_features_subset = torch.index_select(swinT_features_flat, 0, swinT_top_indices.view(-1))

        # Apply attention only on the subset
        attended_features_subset, _ = self.attention(down_features_subset, swinT_features_subset, swinT_features_subset)
        #attended_features_subset, _ = self.attention(cnn_features_subset, swinT_features_subset, swinT_features_subset)

        # Scatter back the attended values to original size tensor
        attended_features = down_features_flat.clone()
        attended_features.index_copy_(0, top_indices.view(-1), attended_features_subset)
        #attended_features.index_copy_(0, cnn_top_indices.view(-1), attended_features_subset)

        # Reshape to [B, C, H, W]
        attended_features = attended_features.permute(1, 2, 0).view_as(downsampling_features)

        # Combine the attended features and original features using the gate
        # out = gate_values * attended_features + (1 - gate_values) * downsampling_features

        return attended_features
