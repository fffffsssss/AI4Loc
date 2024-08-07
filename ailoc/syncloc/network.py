import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import time

import ailoc.common
import ailoc.syncloc


class Out_Head(nn.Module):
    """
    output module
    """

    def __init__(self, c_input, kernel_size=3):
        super().__init__()
        self.res_conv = ailoc.syncloc.Residual(
            nn.Sequential(
                ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                           out_channels=c_input,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=kernel_size // 2),
                ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                           out_channels=c_input,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=kernel_size // 2),
                ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                           out_channels=c_input,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=kernel_size // 2),
            )
        )

        self.p_out = nn.Sequential(ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                                              out_channels=c_input,
                                                              kernel_size=kernel_size,
                                                              stride=1,
                                                              padding=kernel_size // 2),
                                   ailoc.syncloc.Conv2d(in_channels=c_input,
                                                         out_channels=1,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0))
        nn.init.constant_(self.p_out[1][0].bias, -6.)  # -6

        self.xyzph_out = nn.Sequential(ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                                                  out_channels=c_input,
                                                                  kernel_size=kernel_size,
                                                                  stride=1,
                                                                  padding=kernel_size // 2),
                                       ailoc.syncloc.Conv2d(in_channels=c_input,
                                                             out_channels=4,
                                                             kernel_size=1,
                                                             stride=1,
                                                             padding=0))
        nn.init.zeros_(self.xyzph_out[1][0].bias)

        self.xyzphs_out = nn.Sequential(ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                                                   out_channels=c_input,
                                                                   kernel_size=kernel_size,
                                                                   stride=1,
                                                                   padding=kernel_size // 2),
                                        ailoc.syncloc.Conv2d(in_channels=c_input,
                                                              out_channels=4,
                                                              kernel_size=1,
                                                              stride=1,
                                                              padding=0))
        nn.init.zeros_(self.xyzphs_out[1][0].bias)

        self.bg_out = nn.Sequential(ailoc.syncloc.Conv2d_GELU(in_channels=c_input,
                                                               out_channels=c_input,
                                                               kernel_size=kernel_size,
                                                               stride=1,
                                                               padding=kernel_size // 2),
                                    ailoc.syncloc.Conv2d(in_channels=c_input,
                                                          out_channels=1,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0))
        nn.init.zeros_(self.bg_out[1][0].bias)

    def forward(self, x):
        x = self.res_conv(x)
        p = self.p_out(x)
        xyzph = self.xyzph_out(x)
        xyzphs = self.xyzphs_out(x)
        bg = self.bg_out(x)

        # return outputs
        return p, xyzph, xyzphs, bg


class U_NeXt(nn.Module):
    def __init__(self,
                 c_input=1,
                 c_output=48,
                 n_stages=2,
                 kernel_size=3,):

        super().__init__()
        self.c_input = c_input
        self.c_output = c_output
        self.n_stages = n_stages
        self.kernel_size = kernel_size

        # first are two warm up layers
        self.input_head = nn.Sequential(
            ailoc.syncloc.Conv2d(c_input, c_output, 3, 1, 3//2),
            ailoc.syncloc.Conv2d_GELU(c_output, c_output, 3, 1, 3//2))

        #  down sampling blocks
        curr_c = c_output
        self.down_sample = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_sample.append(ailoc.syncloc.DownSampleBlock(curr_c, curr_c * 2, kernel_size))
            curr_c *= 2

        # up sampling blocks
        self.up_sample = nn.ModuleList()
        for i in range(n_stages):
            # add a heavy tail to the first up sampling block
            tail = None
            if i == 0:
                tail = []
                for j in range(6):
                    tail.append(
                        ailoc.syncloc.ConvNextBlock(curr_c//2, curr_c//2, kernel_size),
                    )
                tail = nn.Sequential(*tail)
            self.up_sample.append(ailoc.syncloc.UpSampleBlock(curr_c, curr_c // 2, kernel_size, tail))
            curr_c //= 2

    def forward(self, x):
        x = self.input_head(x)

        skip_list = []
        for i in range(self.n_stages):
            skip_list.append(x)
            x = self.down_sample[i](x)

        for i in range(self.n_stages):
            skip_x = skip_list.pop()
            x = self.up_sample[i](x, skip_x)

        return x


class U_NeXt_v2(nn.Module):
    def __init__(self,
                 c_input=1,
                 c_output=48,
                 n_stages=2,
                 kernel_size=3,):

        super().__init__()
        self.c_input = c_input
        self.c_output = c_output
        self.n_stages = n_stages
        self.kernel_size = kernel_size

        # first are two warm up layers
        self.input_head = nn.Sequential(
            ailoc.syncloc.Conv2d(c_input, c_output, 3, 1, 3//2),
            ailoc.syncloc.Conv2d_GELU(c_output, c_output, 3, 1, 3//2))

        #  down sampling blocks
        curr_c = c_output
        self.down_sample = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_sample.append(ailoc.syncloc.DownSampleBlock_v2(curr_c, curr_c * 2, kernel_size))
            curr_c *= 2

        # up sampling blocks
        self.up_sample = nn.ModuleList()
        for i in range(n_stages):
            self.up_sample.append(ailoc.syncloc.UpSampleBlock_v2(curr_c, curr_c // 2, kernel_size))
            curr_c //= 2

    def forward(self, x):
        x = self.input_head(x)

        skip_list = []
        for i in range(self.n_stages):
            skip_list.append(x)
            x = self.down_sample[i](x)

        for i in range(self.n_stages):
            skip_x = skip_list.pop()
            x = self.up_sample[i](x, skip_x)

        return x


class SyncLocNet(nn.Module):
    """
    The TransLocNet consists of a U-NeXt based feature extraction module(FEM), an optional temporal
    Transformer module(TTM) and an output head. The output representation is inspired by the DECODE.
    """

    def __init__(self, temporal_attn=True, attn_length=3, train_context_size=12):
        super().__init__()

        self.temporal_attn = temporal_attn
        self.attn_length = attn_length
        self.train_context_size = train_context_size
        self.n_features = 48

        # feature extraction module
        self.fem = U_NeXt(c_input=1,
                          c_output=self.n_features,
                          n_stages=2,
                          kernel_size=5).cuda()

        # import ailoc.deeploc
        # self.fem = ailoc.deeploc.Unet(n_inp=1,
        #                               n_filters=self.n_features,
        #                               n_stages=2,
        #                               pad=1,
        #                               ker_size=3).cuda()

        # layer norm for transformer based tam
        self.fem_norm = ailoc.syncloc.LayerNorm(self.n_features, data_format="channels_first").cuda()

        # # layer norm for Unext based tam
        # self.fem_norm = ailoc.syncloc.LayerNorm(self.n_features*3, data_format="channels_first").cuda()

        if self.temporal_attn:
            # temporal attention module
            patch_size = 1
            self.tam = ailoc.syncloc.TransformerBlock(seq_length=train_context_size,
                                                       attn_length=attn_length,
                                                       c_input=self.n_features,
                                                       patch_size=patch_size,
                                                       embedding_dim=(patch_size**2)*self.n_features,
                                                       num_layers=1,
                                                       num_heads=1,
                                                       mlp_dim=(patch_size**2)*4*self.n_features,
                                                       dropout_rate=0.0,
                                                       context_dropout=0.5).cuda()

            # # Unext based temporal attention module
            # self.tam = U_NeXt(c_input=self.n_features * 3,
            #                   c_output=self.n_features,
            #                   n_stages=2,
            #                   kernel_size=5).cuda()

            # # Unet based temporal attention module
            # self.tam = ailoc.deeploc.Unet(n_inp=self.n_features * 3,
            #                               n_filters=self.n_features,
            #                               n_stages=2,
            #                               pad=1,
            #                               ker_size=3).cuda()

        self.out_head = Out_Head(c_input=self.n_features, kernel_size=3).cuda()

        self.get_parameter_number()

    def forward(self, x_input):
        """
        network forward propagation.

        Args:
            x_input (torch.Tensor): scaled SMLM images

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): predicted probability map, xyzphoton map,
                xyzphoton uncertainty map, background map
        """

        img_h, img_w = x_input.shape[-2], x_input.shape[-1]

        assert img_h % 4 == 0 and img_w % 4 == 0, 'network structure requires that image size should be multiples of 4'

        # during training and online evaluation, the input should have the shape
        # (training batch size, context_size, height, width), the first and last attn_length//2 frames are
        # used to provide temporal context, which will not be used to compute loss
        if x_input.ndimension() == 4:
            batch_size, context_size = x_input.shape[:2]
            fem_out = self.fem(x_input.reshape([-1, 1, img_h, img_w]))
            if self.temporal_attn:
                # transformer based tam
                fem_out = fem_out.reshape([batch_size, context_size, self.n_features, img_h, img_w])

                # # Unext based tam, can only deal with 3 frames as a local context
                # fem_out = fem_out.reshape([batch_size, context_size, -1, img_h, img_w])
                # zeros = torch.zeros_like(fem_out[:, :1])
                # h_t1 = fem_out
                # h_t0 = torch.cat([zeros, fem_out[:, :-1]], dim=1)
                # h_t2 = torch.cat([fem_out[:, 1:], zeros], dim=1)
                # fem_out = torch.cat([h_t0, h_t1, h_t2], dim=2)[:, 1:-1].reshape(-1, self.n_features * 3, img_h, img_w)

        # when analyzing experimental data, the input dimension is 3, (anlz_batch_size, height, width), each batch
        # is padded with attn_length//2 frames at the beginning and end to provide temporal context
        elif x_input.ndimension() == 3:
            anlz_batch_size = x_input.shape[0]
            fem_out = self.fem(x_input[:, None])
            if self.temporal_attn:
                # transformer based tam
                fem_out = fem_out.reshape([1, anlz_batch_size, self.n_features, img_h, img_w])

                # # Unext based tam, can only deal with 3 frames as a local context
                # # create a zero output with the shape (1, 48, height, width) to pad the head and tail
                # zeros = torch.zeros_like(fem_out[:1])
                # # the fm_out has the shape (analysis batch size+2, 48, height, width)
                # h_t1 = fem_out
                # # pad zero to the head and discard the tail frame output of the batch
                # h_t0 = torch.cat([zeros, fem_out], 0)[:-1]
                # # pad zero to the tail and discard the head frame output of the batch
                # h_t2 = torch.cat([fem_out, zeros], 0)[1:]
                # # reuse the features of each frame to build the local context with the shape
                # # (analysis batch size+2, 144, height, width), discard the head and tail
                # fem_out = torch.cat([h_t0, h_t1, h_t2], 1)[1:-1]
        else:
            raise ValueError('The input dimension is not supported.')

        # layer normalization
        fem_out = self.fem_norm(fem_out)

        if self.temporal_attn:
            # transformer based tam, since each traning/analysis batch is padded with attn_length//2 frames
            # at the beginning and end, we only need the prediction of the middle frames
            tam_out = self.tam(fem_out)
            if self.attn_length//2 > 0:
                tam_out = tam_out[:, self.attn_length//2: -(self.attn_length//2)].reshape([-1, self.n_features, img_h, img_w])
            else:
                tam_out = tam_out.reshape([-1, self.n_features, img_h, img_w])

            # # Unext based tam, can only deal with 3 frames as a local context
            # tam_out = self.tam(fem_out)

            p, xyzph, xyzphs, bg = self.out_head(tam_out)
        else:
            p, xyzph, xyzphs, bg = self.out_head(fem_out)

        p_pred = torch.sigmoid(torch.clamp(p, min=-16, max=16))[:, 0]  # probability
        xyzph_pred = xyzph
        # output xy range is (-1, 1), the training sampling range is (-0.5,0.5)
        xyzph_pred[:, :2] = torch.tanh(xyzph_pred[:, :2])
        # output z range is (-2, 2), the training sampling range is in (-1,1)
        xyzph_pred[:, 2] = torch.tanh(xyzph_pred[:, 2]) * 2
        # output photon range is (0, 1), the training sampling range is in (0,1)
        xyzph_pred[:, 3] = torch.sigmoid(xyzph_pred[:, 3])
        # scale the uncertainty and add epsilon, the output range becomes (0.0001, 3.0001),
        # maybe can use RELU for unlimited upper range and stable gradient
        xyzph_sig_pred = torch.sigmoid(xyzphs) * 3 + 0.0001
        # output bg range is (0, 1), the training sampling range is in (0,1)
        bg_pred = torch.sigmoid(bg[:, 0])  # bg

        return p_pred, xyzph_pred, xyzph_sig_pred, bg_pred

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(1, self.train_context_size, 64, 64).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(1000):
            self.forward(dummy_input)
        print(f'Average forward time: {(time.time()-t0)/1000:.4f} s')
