import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# Conditionally uses torch's checkpoint functionality if it is enabled in the opt file.
from torch.nn import SiLU

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

        return self.normalize(x5 * .2 + x)


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
        self.residual_mult = nn.Parameter(torch.FloatTensor([.1]))

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

        return self.normalize(out * self.residual_mult + x)


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
