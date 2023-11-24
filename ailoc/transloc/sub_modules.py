import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import ailoc.transloc


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, attn_mask=None):
        x = x + self.fn(x, attn_mask) if isinstance(self.fn, ailoc.transloc.SelfAttention) else x + self.fn(x)
        return x


class MultiScaleConv2D(nn.Module):
    def __init__(self, c_input, c_output, kernel_list=(3, 5, 7)):
        super().__init__()
        num_kernel = len(kernel_list)
        assert c_output % num_kernel == 0, "c_output must be divisible by num_kernel"
        self.kernel_list = kernel_list
        self.c_output_per_kernel = c_output // num_kernel
        self.c_input = c_input
        self.conv_list = nn.ModuleList()
        self.activation = nn.ELU()

        for kernel_size in kernel_list:
            self.conv_list.append(nn.Conv2d(c_input, self.c_output_per_kernel, kernel_size, padding=kernel_size // 2))

        for conv in self.conv_list:
            if isinstance(conv, nn.Conv2d):
                nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        output_list = []
        for conv in self.conv_list:
            output_list.append(conv(x))
        output = self.activation(torch.cat(output_list, dim=1))
        return output


class MultiScaleDilatedConv2D(nn.Module):
    def __init__(self, c_input, c_output, dilated_list=(1, 2, 3)):
        super().__init__()
        num_dilated = len(dilated_list)
        assert c_output % num_dilated == 0, "c_output must be divisible by num_dilated"
        self.dilated_list = dilated_list
        self.c_output_per_dilation = c_output // num_dilated
        self.c_input = c_input
        self.conv_list = nn.ModuleList()
        self.activation = nn.ELU()

        for dilation in dilated_list:
            self.conv_list.append(nn.Conv2d(in_channels=c_input,
                                            out_channels=self.c_output_per_dilation,
                                            kernel_size=3,
                                            stride=1,
                                            padding=dilation,
                                            dilation=dilation,))

        for conv in self.conv_list:
            if isinstance(conv, nn.Conv2d):
                nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        output_list = []
        for conv in self.conv_list:
            output_list.append(conv(x))
        output = self.activation(torch.cat(output_list, dim=1))
        return output


class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups),
        )
        nn.init.kaiming_normal_(self[0].weight, mode='fan_in', nonlinearity='relu')


class Conv2d_ELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d_ELU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ELU()
        )
        nn.init.kaiming_normal_(self[0].weight, mode='fan_in', nonlinearity='relu')


class Conv2d_GELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GELU()
        )
        nn.init.kaiming_normal_(self[0].weight, mode='fan_in', nonlinearity='relu')


class TwoConv2d_ELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TwoConv2d_ELU, self).__init__(
            Conv2d_ELU(in_channels, out_channels, kernel_size, stride, padding),
            Conv2d_ELU(out_channels, out_channels, kernel_size, stride, padding),
        )


class TwoConv2d_GELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TwoConv2d_GELU, self).__init__(
            Conv2d_ELU(in_channels, out_channels, kernel_size, stride, padding),
            Conv2d_ELU(out_channels, out_channels, kernel_size, stride, padding),
        )


class UpSampleNN2d_ELU(nn.Sequential):
    def __init__(self, scale_factor):
        super(UpSampleNN2d_ELU, self).__init__(
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            nn.ELU()
        )


class UpSampleNN2d_GELU(nn.Sequential):
    def __init__(self, scale_factor):
        super().__init__(
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            nn.GELU()
        )


class UpSampleBL2d_ELU(nn.Sequential):
    def __init__(self, scale_factor):
        super(UpSampleBL2d_ELU, self).__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.ELU()
        )


class DownSampleBlock(nn.Sequential):
    """
    for down-sample stage, the first layer half the spatial size and
    keeps the number of channels the same, then convolve to double the number of channels,
    finally convolve again at doubled channels
    """

    # def __init__(self, in_channels, out_channels, kernel_size):
    #     super(DownSampleBlock, self).__init__(
    #         Conv2d_GELU(in_channels, in_channels, 2, 2, 0),
    #         Conv2d_GELU(in_channels, out_channels, kernel_size, 1, kernel_size//2),
    #         Residual(Conv2d_GELU(out_channels, out_channels, kernel_size, 1, kernel_size//2)),
    #     )

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownSampleBlock, self).__init__(
            LayerNorm(in_channels, data_format="chw", elementwise_affine=False),
            Conv2d(in_channels, in_channels, 2, 2, 0),
            ConvNextBlock(in_channels, out_channels, kernel_size),
            ConvNextBlock(out_channels, out_channels, kernel_size),
        )


class UpSampleBlock(nn.Module):
    """
    for up-sample stage, the first layer double the spatial size and
    keeps the number of channels the same, then convolve to half the number of channels,
    the third layer receives the concatenation of the previous layer output and
    the skip output of the corresponding down-sample stage and convolve to half the channels,
    the fourth layer convolve again at half channels
    """

    # def __init__(self, in_channels, out_channels, kernel_size, tail=None):
    #     super(UpSampleBlock, self).__init__()
    #     self.layers = nn.ModuleList()
    #     self.layers.append(UpSampleNN2d_GELU(2))
    #     self.layers.append(Conv2d_GELU(in_channels, out_channels, kernel_size, 1, kernel_size // 2))
    #     self.layers.append(Conv2d_GELU(in_channels, out_channels, kernel_size, 1, kernel_size // 2))
    #     self.layers.append(Conv2d_GELU(out_channels, out_channels, kernel_size, 1, kernel_size // 2))
    #     self.tail = tail

    def __init__(self, in_channels, out_channels, kernel_size, tail=None):
        super(UpSampleBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LayerNorm(in_channels, data_format="chw", elementwise_affine=False))
        self.layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        self.layers.append(ConvNextBlock(in_channels, out_channels, kernel_size))
        self.layers.append(ConvNextBlock(in_channels, out_channels, kernel_size))
        self.layers.append(ConvNextBlock(out_channels, out_channels, kernel_size))
        self.tail = tail

    def forward(self, x, x_skip):
        x = self.layers[0](x)  # LayerNorm
        x = self.layers[1](x)  # UpSampleNN2d_GELU
        x = self.layers[2](x)  # ConvNextBlock
        x = torch.cat([x, x_skip], dim=1)
        x = self.layers[3](x)
        x = self.layers[4](x)
        if self.tail is not None:
            x = self.tail(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last", elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first", "chw"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.weight is not None:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = x
            return x
        elif self.data_format == "chw":
            return F.layer_norm(x, normalized_shape=x.shape[1:])


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    # def __init__(self, c_in, c_out, kernel_size=7, layer_scale_init_value=0):
    #     super().__init__()
    #     self.c_in = c_in
    #     self.c_out = c_out
    #     self.dwconv = Conv2d(in_channels=c_in,
    #                          out_channels=c_in,
    #                          kernel_size=kernel_size,
    #                          stride=1,
    #                          padding=kernel_size//2,
    #                          groups=c_in)  # depthwise conv
    #
    #     self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_last")
    #     self.pwconv1 = nn.Linear(c_in, 4 * c_in)  # pointwise/1x1 convs, implemented with linear layers
    #     self.act = nn.GELU()
    #     self.pwconv2 = nn.Linear(4 * c_in, c_out)
    #     self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_out)),
    #                               requires_grad=True) if layer_scale_init_value > 0 else None
    #
    # def forward(self, x):
    #     input = x
    #     x = self.dwconv(x)
    #     # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    #     x = rearrange(x, 'b c h w -> b h w c')
    #     # x = self.norm(x)
    #     x = self.pwconv1(x)
    #     x = self.act(x)
    #     x = self.pwconv2(x)
    #     if self.gamma is not None:
    #         x = self.gamma * x
    #     # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    #     x = rearrange(x, 'b h w c -> b c h w')
    #
    #     if self.c_in == self.c_out:
    #         x = input + x
    #     return x

    def __init__(self, c_in, c_out, kernel_size=7, layer_scale_init_value=0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dwconv = Conv2d(in_channels=c_in,
                             out_channels=c_in,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             groups=c_in)  # depthwise conv

        self.norm = LayerNorm(c_in, data_format="chw", elementwise_affine=False)
        self.pwconv1 = Conv2d(in_channels=c_in, out_channels=4*c_in, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.pwconv2 = Conv2d(in_channels=4*c_in, out_channels=c_out, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_out)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        if self.c_in == self.c_out:
            x = input + x
        return x
