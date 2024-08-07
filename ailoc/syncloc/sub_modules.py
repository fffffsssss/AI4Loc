import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import ailoc.syncloc


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, attn_mask=None):
        x = x + self.fn(x, attn_mask) if isinstance(self.fn, ailoc.syncloc.SelfAttention) else x + self.fn(x)
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
            LayerNorm(in_channels, data_format="channels_first"),
            Conv2d(in_channels, in_channels, 2, 2, 0),
            ConvNextBlock(in_channels, out_channels, kernel_size),
            # ConvNextBlock(out_channels, out_channels, kernel_size),
            # InceptionNextBlock(in_channels, out_channels, band_kernel_size=kernel_size),
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
        self.layers.append(LayerNorm(in_channels, data_format="channels_first"))
        self.layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        # self.layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.layers.append(ConvNextBlock(in_channels, out_channels, kernel_size))
        self.layers.append(ConvNextBlock(in_channels, out_channels, kernel_size))
        # self.layers.append(ConvNextBlock(out_channels, out_channels, kernel_size))
        # self.layers.append(InceptionNextBlock(in_channels, out_channels, band_kernel_size=kernel_size))
        # self.layers.append(InceptionNextBlock(in_channels, out_channels, band_kernel_size=kernel_size))
        self.tail = tail

    def forward(self, x, x_skip):
        x = self.layers[0](x)  # LayerNorm
        x = self.layers[1](x)  # UpSampleNN2d_GELU
        x = self.layers[2](x)  # ConvNextBlock
        x = torch.cat([x, x_skip], dim=1)
        x = self.layers[3](x)
        # x = self.layers[4](x)
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
            u = x.mean(-3, keepdim=True)
            s = (x - u).pow(2).mean(-3, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.weight is not None:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = x
            return x
        elif self.data_format == "chw":
            return F.layer_norm(x, normalized_shape=x.shape[-3:])


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, c_in, c_out, kernel_size=7, layer_scale_init_value=0, version=1):
        super().__init__()
        self.version = version
        self.c_in = c_in
        self.c_out = c_out
        self.dwconv = Conv2d(in_channels=c_in,
                             out_channels=c_in,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             groups=c_in)  # depthwise conv

        # self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_last")
        # self.norm = LayerNorm(c_in, data_format="chw", elementwise_affine=False)
        self.pwconv1 = nn.Linear(c_in, 4 * c_in)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * c_in, c_out)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_out)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        # x = rearrange(x, 'b c h w -> b h w c')
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        # x = rearrange(x, 'b h w c -> b c h w')
        if self.c_in == self.c_out:
            x = input + x
        return x

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
    #     # self.norm = LayerNorm(c_in, data_format="chw", elementwise_affine=False)
    #     self.pwconv1 = Conv2d(in_channels=c_in, out_channels=4*c_in, kernel_size=1, stride=1, padding=0)
    #     self.act = nn.GELU()
    #     self.pwconv2 = Conv2d(in_channels=4*c_in, out_channels=c_out, kernel_size=1, stride=1, padding=0)
    #     self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_out)),
    #                               requires_grad=True) if layer_scale_init_value > 0 else None
    #
    # def forward(self, x):
    #     input = x
    #     x = self.dwconv(x)
    #     # x = self.norm(x)
    #     x = self.pwconv1(x)
    #     x = self.act(x)
    #     x = self.pwconv2(x)
    #     if self.gamma is not None:
    #         x = self.gamma * x
    #     if self.c_in == self.c_out:
    #         x = input + x
    #     return x

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
    #     self.conv1 = Conv2d(c_in, c_out, 3, 1, 1)
    #     self.act = nn.GELU()
    #
    # def forward(self, x):
    #     input = x
    #     x = self.dwconv(x)
    #     x = self.conv1(x)
    #     x = self.act(x)
    #     if self.c_in == self.c_out:
    #         x = input + x
    #     return x


class InceptionNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.square_kernel_size = square_kernel_size
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio

        self.gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(self.gc,
                                   self.gc,
                                   square_kernel_size,
                                   padding=square_kernel_size // 2,
                                   groups=self.gc)
        self.dwconv_w = nn.Conv2d(self.gc,
                                  self.gc,
                                  kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2),
                                  groups=self.gc)
        self.dwconv_h = nn.Conv2d(self.gc,
                                  self.gc,
                                  kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0),
                                  groups=self.gc)
        self.split_indexes = (in_channels - 3 * self.gc, self.gc, self.gc, self.gc)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        # self.pwconv1 = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
        # self.pwconv2 = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        input = x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1,)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.in_channels == self.out_channels:
            x = input + x
        return x


class DownSampleBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = LayerNorm(in_channels, data_format="chw", elementwise_affine=False)
        self.downconv = Conv2d(in_channels, in_channels, 2, 2, 0)
        self.dwconv = Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, groups=in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.downconv(x)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class UpSampleBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = LayerNorm(in_channels, data_format="chw", elementwise_affine=False)
        self.upconv = nn.UpsamplingNearest2d(scale_factor=2)
        self.dwconv = Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2, groups=in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * out_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.norm(x)
        x = self.upconv(x)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x += x_skip
        return x


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
