import functools
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from math import sqrt


# Conditionally uses torch's checkpoint functionality if it is enabled in the opt file.
from guided_diffusion.nn import linear, timestep_embedding
from guided_diffusion.unet import TimestepEmbedSequential, TimestepBlock


def checkpoint(fn, *args):
    return torch.utils.checkpoint.checkpoint(fn, *args)


def sequential_checkpoint(fn, partitions, *args):
    return torch.utils.checkpoint.checkpoint_sequential(fn, partitions, *args)


def opt_get(opt, keys, default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdims=True) + epsilon)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return TimestepEmbedSequential(*layers)


def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


class ResidualBlock(nn.Module):
    '''Residual block with BN
    ---Conv-BN-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN2 = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.BN1(self.conv1(x)))
        out = self.BN2(self.conv2(out))
        return identity + out

class ResidualBlockSpectralNorm(nn.Module):
    '''Residual block with Spectral Normalization.
    ---SpecConv-ReLU-SpecConv-+-
     |________________|
    '''

    def __init__(self, nf, total_residual_blocks):
        super(ResidualBlockSpectralNorm, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.conv2 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        initialize_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class ResidualBlockGN(nn.Module):
    '''Residual block with GroupNorm
    ---Conv-GN-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlockGN, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1 = nn.GroupNorm(8, nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN2 = nn.GroupNorm(8, nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.BN1(self.conv1(x)))
        out = self.BN2(self.conv2(out))
        return identity + out


class PixelUnshuffle(nn.Module):
    def __init__(self, reduction_factor):
        super(PixelUnshuffle, self).__init__()
        self.r = reduction_factor

    def forward(self, x):
        (b, f, w, h) = x.shape
        x = x.contiguous().view(b, f, w // self.r, self.r, h // self.r, self.r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, f * (self.r ** 2), w // self.r, h // self.r)
        return x


# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input)


''' Convenience class with Conv->GroupNorm->LeakyReLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvGnLelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, num_groups=8, weight_init_factor=1):
        super(ConvGnLelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.gn = nn.GroupNorm(num_groups, filters_out)
        else:
            self.gn = None
        if activation:
            self.lelu = nn.LeakyReLU(negative_slope=.2)
        else:
            self.lelu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=.1, mode='fan_out',
                                        nonlinearity='leaky_relu' if self.lelu else 'linear')
                m.weight.data *= weight_init_factor
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.gn:
            x = self.gn(x)
        if self.lelu:
            return self.lelu(x)
        else:
            return x


''' Convenience class with Conv->BN->SiLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvGnSilu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, num_groups=8, weight_init_factor=1):
        super(ConvGnSilu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.gn = nn.GroupNorm(num_groups, filters_out)
        else:
            self.gn = None
        if activation:
            self.silu = SiLU()
        else:
            self.silu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if self.silu else 'linear')
                m.weight.data *= weight_init_factor
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.gn:
            x = self.gn(x)
        if self.silu:
            return self.silu(x)
        else:
            return x


# Simple way to chain multiple conv->act->norms together in an intuitive way.
class MultiConvBlock(nn.Module):
    def __init__(self, filters_in, filters_mid, filters_out, kernel_size, depth, scale_init=1, norm=False, weight_init_factor=1):
        assert depth >= 2
        super(MultiConvBlock, self).__init__()
        self.noise_scale = nn.Parameter(torch.full((1,), fill_value=.01))
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor)] +
                                     [ConvBnLelu(filters_mid, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor) for i in range(depth - 2)] +
                                     [ConvBnLelu(filters_mid, filters_out, kernel_size, activation=False, norm=False, bias=False, weight_init_factor=weight_init_factor)])
        self.scale = nn.Parameter(torch.full((1,), fill_value=scale_init, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is not None:
            noise = noise * self.noise_scale
            x = x + noise
        for m in self.bnconvs:
            x = m.forward(x)
        return x * self.scale + self.bias


# Block that upsamples 2x and reduces incoming filters by 2x. It preserves structure by taking a passthrough feed
# along with the feature representation.
class ExpansionBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=True)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=False, norm=True)
        self.conjoin = block(filters_out*2, filters_out, kernel_size=3, bias=False, activation=True, norm=False)
        self.process = block(filters_out, filters_out, kernel_size=3, bias=False, activation=True, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.process(x)


# Block that upsamples 2x and reduces incoming filters by 2x. It preserves structure by taking a passthrough feed
# along with the feature representation.
# Differs from ExpansionBlock because it performs all processing in 2xfilter space and decimates at the last step.
class ExpansionBlock2(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock2, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=True)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=False, norm=True)
        self.conjoin = block(filters_out*2, filters_out*2, kernel_size=3, bias=False, activation=True, norm=False)
        self.reduce = block(filters_out*2, filters_out, kernel_size=3, bias=False, activation=True, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.reduce(x)


# Similar to ExpansionBlock2 but does not upsample.
class ConjoinBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, filters_pt=None, block=ConvGnSilu, norm=True):
        super(ConjoinBlock, self).__init__()
        if filters_out is None:
            filters_out = filters_in
        if filters_pt is None:
            filters_pt = filters_in
        self.process = block(filters_in + filters_pt, filters_in + filters_pt, kernel_size=3, bias=False, activation=True, norm=norm)
        self.decimate = block(filters_in + filters_pt, filters_out, kernel_size=1, bias=False, activation=False, norm=norm)

    def forward(self, input, passthrough):
        x = torch.cat([input, passthrough], dim=1)
        x = self.process(x)
        return self.decimate(x)


# Designed explicitly to join a mainline trunk with reference data. Implemented as a residual branch.
class ReferenceJoinBlock(nn.Module):
    def __init__(self, nf, residual_weight_init_factor=1, block=ConvGnLelu, final_norm=False, kernel_size=3, depth=3, join=True):
        super(ReferenceJoinBlock, self).__init__()
        self.branch = MultiConvBlock(nf * 2, nf + nf // 2, nf, kernel_size=kernel_size, depth=depth,
                                     scale_init=residual_weight_init_factor, norm=False,
                                     weight_init_factor=residual_weight_init_factor)
        if join:
            self.join_conv = block(nf, nf, kernel_size=kernel_size, norm=final_norm, bias=False, activation=True)
        else:
            self.join_conv = None

    def forward(self, x, ref):
        joined = torch.cat([x, ref], dim=1)
        branch = self.branch(joined)
        if self.join_conv is not None:
            return self.join_conv(x + branch), torch.std(branch)
        else:
            return x + branch, torch.std(branch)


# Basic convolutional upsampling block that uses interpolate.
class UpconvBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu, norm=True, activation=True, bias=False):
        super(UpconvBlock, self).__init__()
        self.process = block(filters_in, filters_out, kernel_size=3, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.process(x)


# Scales an image up 2x and performs intermediary processing. Designed to be the final block in an SR network.
class FinalUpsampleBlock2x(nn.Module):
    def __init__(self, nf, block=ConvGnLelu, out_nc=3, scale=2):
        super(FinalUpsampleBlock2x, self).__init__()
        if scale == 2:
            self.chain = nn.Sequential(block(nf, nf, kernel_size=3, norm=False, activation=True, bias=True),
                                       UpconvBlock(nf, nf // 2, block=block, norm=False, activation=True, bias=True),
                                       block(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True),
                                       block(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False))
        else:
            self.chain = nn.Sequential(block(nf, nf, kernel_size=3, norm=False, activation=True, bias=True),
                                       UpconvBlock(nf, nf, block=block, norm=False, activation=True, bias=True),
                                       block(nf, nf, kernel_size=3, norm=False, activation=False, bias=True),
                                       UpconvBlock(nf, nf // 2, block=block, norm=False, activation=True, bias=True),
                                       block(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True),
                                       block(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False))

    def forward(self, x):
        return self.chain(x)


class ResidualDenseBlock(TimestepBlock):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32, embedding=False, init_weight=.1):
        super(ResidualDenseBlock, self).__init__()
        self.embedding = embedding
        if embedding:
            self.first_conv = ConvGnLelu(mid_channels, mid_channels, activation=True, norm=False, bias=True)
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    mid_channels*4,
                    mid_channels,
                ),
            )
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i + 1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(4):
            default_init_weights(getattr(self, f'conv{i + 1}'), init_weight)
        default_init_weights(self.conv5, 0)

        self.normalize = nn.GroupNorm(num_groups=8, num_channels=mid_channels)

    def forward(self, x, emb):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.embedding:
            x0 = self.first_conv(x)
            emb_out = self.emb_layers(emb).type(x0.dtype)
            while len(emb_out.shape) < len(x0.shape):
                emb_out = emb_out[..., None]
            x0 = x0 + emb_out
        else:
            x0 = x
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return self.normalize(x5 + x)


class RRDB(TimestepBlock):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels, embedding=True)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        self.normalize = nn.GroupNorm(num_groups=8, num_channels=mid_channels)

    def forward(self, x, emb):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x, emb)
        out = self.rdb2(out, emb)
        out = self.rdb3(out, emb)

        return self.normalize(out + x)


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 body_block=RRDB,
                 ):
        super(RRDBNet, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # The diffusion RRDB starts with a full resolution image and downsamples into a .25 working space
        self.input_blocks = nn.Sequential(ConvGnLelu(in_channels, mid_channels, kernel_size=7, stride=2, activation=True, norm=False, bias=True),
                                          ConvGnLelu(mid_channels, mid_channels, kernel_size=3, stride=2, activation=True, norm=False, bias=True))

        # Guided diffusion uses a time embedding.
        time_embed_dim = mid_channels * 4
        self.time_embed = nn.Sequential(
            linear(mid_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.body = make_layer(
            body_block,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)

        self.conv_body = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        self.conv_up3 = None
        self.conv_hr = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.mid_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.normalize = nn.GroupNorm(num_groups=8, num_channels=self.mid_channels)

        for m in [
            self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr
        ]:
            if m is not None:
                default_init_weights(m, 1.0)
        default_init_weights(self.conv_last, 0)

    def forward(self, x, timesteps, low_res=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.mid_channels))

        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)

        feat = self.input_blocks(x)
        for bl in self.body:
            feat = checkpoint(bl, feat, emb)
        feat = feat[:, :self.mid_channels]
        body_feat = self.conv_body(feat)
        feat = self.normalize(feat + body_feat)

        # upsample
        out = self.lrelu(
            self.normalize(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest'))))
        out = self.lrelu(
            self.normalize(self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest'))))
        out = self.conv_last(self.normalize(self.lrelu(self.conv_hr(out))))

        return out


if __name__ == '__main__':
    model = RRDBNet(6,6)
    x = torch.randn(1,3,128,128)
    l = torch.randn(1,3,32,32)
    t = torch.LongTensor([555])
    y = model(x, t, l)
    print(y.shape, y.mean(), y.std(), y.min(), y.max())
