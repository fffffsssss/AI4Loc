import torch
import torch.nn as nn
import torch.nn.functional as F
import thop

import ailoc.common
import ailoc.transloc


class Outnet(nn.Module):
    """
    output module
    """

    def __init__(self, n_input, pad=1, ker_size=3):
        super().__init__()

        self.p_out1 = nn.Conv2d(in_channels=n_input, out_channels=n_input, kernel_size=ker_size,
                                padding=pad).cuda()
        self.p_out2 = nn.Conv2d(in_channels=n_input, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzph_out1 = nn.Conv2d(in_channels=n_input, out_channels=n_input, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.xyzph_out2 = nn.Conv2d(in_channels=n_input, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.p_out2.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzph_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzph_out2.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.zeros_(self.xyzph_out2.bias)

        self.xyzphs_out1 = nn.Conv2d(in_channels=n_input, out_channels=n_input, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzphs_out2 = nn.Conv2d(in_channels=n_input, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzphs_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzphs_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzphs_out2.bias)

        self.bg_out1 = nn.Conv2d(in_channels=n_input, out_channels=n_input, kernel_size=ker_size,
                                 padding=pad).cuda()
        self.bg_out2 = nn.Conv2d(in_channels=n_input, out_channels=1, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.bg_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.bg_out2.bias)

    def forward(self, x):

        # outputs = {}

        p = F.elu(self.p_out1(x))
        # outputs['p'] = self.p_out2(p)
        p = self.p_out2(p)

        xyzph = F.elu(self.xyzph_out1(x))
        # outputs['xyzph'] = self.xyzph_out2(xyzph)
        xyzph = self.xyzph_out2(xyzph)

        xyzphs = F.elu(self.xyzphs_out1(x))
        # outputs['xyzph_sig'] = self.xyzphs_out2(xyzphs)
        xyzphs = self.xyzphs_out2(xyzphs)

        bg = F.elu(self.bg_out1(x))
        # outputs['bg'] = self.bg_out2(bg)
        bg = self.bg_out2(bg)

        # return outputs
        return p, xyzph, xyzphs, bg


class Out_Head(nn.Module):
    """
    output module
    """

    def __init__(self, c_input, kernel_size=3):
        super().__init__()
        self.norm = ailoc.transloc.LayerNorm(c_input, data_format="chw", elementwise_affine=False)

        self.p_out = nn.Sequential(ailoc.transloc.Conv2d_GELU(in_channels=c_input,
                                                              out_channels=c_input,
                                                              kernel_size=kernel_size,
                                                              stride=1,
                                                              padding=kernel_size // 2),
                                   ailoc.transloc.Conv2d(in_channels=c_input,
                                                         out_channels=1,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0))
        nn.init.constant_(self.p_out[1][0].bias, -6.)  # -6

        self.xyzph_out = nn.Sequential(ailoc.transloc.Conv2d_GELU(in_channels=c_input,
                                                                  out_channels=c_input,
                                                                  kernel_size=kernel_size,
                                                                  stride=1,
                                                                  padding=kernel_size // 2),
                                       ailoc.transloc.Conv2d(in_channels=c_input,
                                                             out_channels=4,
                                                             kernel_size=1,
                                                             stride=1,
                                                             padding=0))
        nn.init.zeros_(self.xyzph_out[1][0].bias)

        self.xyzphs_out = nn.Sequential(ailoc.transloc.Conv2d_GELU(in_channels=c_input,
                                                                   out_channels=c_input,
                                                                   kernel_size=kernel_size,
                                                                   stride=1,
                                                                   padding=kernel_size // 2),
                                        ailoc.transloc.Conv2d(in_channels=c_input,
                                                              out_channels=4,
                                                              kernel_size=1,
                                                              stride=1,
                                                              padding=0))
        nn.init.zeros_(self.xyzphs_out[1][0].bias)

        self.bg_out = nn.Sequential(ailoc.transloc.Conv2d_GELU(in_channels=c_input,
                                                               out_channels=c_input,
                                                               kernel_size=kernel_size,
                                                               stride=1,
                                                               padding=kernel_size // 2),
                                    ailoc.transloc.Conv2d(in_channels=c_input,
                                                          out_channels=1,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0))
        nn.init.zeros_(self.bg_out[1][0].bias)

    def forward(self, x):
        x = self.norm(x)
        p = self.p_out(x)
        xyzph = self.xyzph_out(x)
        xyzphs = self.xyzphs_out(x)
        bg = self.bg_out(x)

        # return outputs
        return p, xyzph, xyzphs, bg


class Unet(nn.Module):
    """
    used for frame analysis module and temporal context module
    """

    def __init__(self, n_input, n_filters=48, n_stages=2, pad=1, ker_size=3):
        super().__init__()

        curr_n = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()

        self.layer_path.append(
            nn.Conv2d(in_channels=n_input, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_n *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            # self.layer_path.append(nn.UpsamplingBilinear2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_n //= 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n * 2, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for m in self.layer_path:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        n_l = 0
        x_bridged = []

        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1
        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(4):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1)  # concatenate

        return x


class MultiScaleTransUnet(nn.Module):
    """
    for each convolution, use multiple kernel sizes and concatenate the outputs
    """

    def __init__(self,
                 c_input,
                 c_output=48,
                 hw_input=64,
                 n_stages=2,
                 kernel_list=(3, 5, 7),
                 parallel_stages=1):

        super().__init__()
        self.n_stages = n_stages
        self.parallel_stages = parallel_stages
        self.kernel_list = kernel_list
        self.n_kernel_size = len(kernel_list)

        assert n_stages - parallel_stages >= 0, 'n_stages must be larger than parallel_stages'
        assert c_output % self.n_kernel_size == 0, 'c_output must be divisible by n_kernel_size'
        curr_c = c_output // self.n_kernel_size

        # first are two warm up layers with different kernel sizes
        # self.warm_up_head = nn.ModuleList()
        # for i in range(self.n_kernel_size):
        #     self.warm_up_head.append(ailoc.transloc.TwoConv2d_ELU(c_input, curr_c, kernel_list[i], 1, kernel_list[i]//2))
        self.warm_up_head = nn.Sequential(ailoc.transloc.MultiScaleConv2D(c_input, c_output, kernel_list),
                                          ailoc.transloc.MultiScaleConv2D(c_output, c_output, kernel_list))


        # # parallel down sampling blocks with different kernel sizes
        # if self.parallel_stages > 0:
        #     self.parallel_down_sample = nn.ModuleList()
        #     for i in range(self.parallel_stages):
        #         for j in range(self.n_kernel_size):
        #             self.parallel_down_sample.append(ailoc.transloc.DownSampleBlock(curr_c, curr_c * 2, kernel_list[j]))
        #         curr_c *= 2

        # unified down sampling blocks with kernel size 3
        curr_c *= self.n_kernel_size
        if self.n_stages-self.parallel_stages > 0:
            self.down_sample = nn.ModuleList()
            for i in range(self.n_stages-self.parallel_stages):
                self.down_sample.append(ailoc.transloc.DownSampleBlock(curr_c, curr_c * 2, 3))
                curr_c *= 2

        # unified up sampling blocks with kernel size 3
        self.up_sample = nn.ModuleList()
        for i in range(n_stages):
            self.up_sample.append(ailoc.transloc.UpSampleBlock(curr_c, curr_c // 2, 3))
            curr_c //= 2

        # # n_stages skip connections using transformer blocks
        # self.skip_transformer = nn.ModuleList()
        # for i in range(n_stages):
        #     self.skip_transformer.append(ailoc.transloc.TransformerBlock(seq_length=curr_c,
        #                                                                  hw_input=int(hw_input/(2**i)),
        #                                                                  embedding_dim=16*16,
        #                                                                  num_layers=2,
        #                                                                  num_heads=8,
        #                                                                  mlp_dim=16*16*4,
        #                                                                  input_dropout_rate=0,
        #                                                                  attn_dropout_rate=0,
        #                                                                  ))
        #     curr_c *= 2

        self.trans_block = nn.ModuleList()
        patch_size = 16
        self.trans_block.append(
            ailoc.transloc.TransformerBlock(seq_length=int((hw_input/(2**n_stages)/patch_size)**2),
                                            hw_input=int(hw_input/(2**n_stages)),
                                            # channel_in=c_output*(2**n_stages),
                                            # embedding_channels=16,
                                            embedding_dim=patch_size ** 2,
                                            num_layers=2,
                                            num_heads=8,
                                            mlp_dim=(patch_size ** 2) * 4,
                                            input_dropout_rate=0,
                                            attn_dropout_rate=0,
                                            ))

    def forward(self, x):
        # warm_up_out = []
        # for i in range(self.n_kernel_size):
        #     warm_up_out.append(self.warm_up_head[i](x))

        x = self.warm_up_head(x)


        skip_list = []
        # parallel_out = warm_up_out
        # if self.parallel_stages > 0:
        #     for i in range(self.parallel_stages):
        #         skip_list.append(torch.cat(parallel_out, 1))
        #         for j in range(self.n_kernel_size):
        #             parallel_out[j] = self.parallel_down_sample[i * self.n_kernel_size + j](parallel_out[j])
        # x = torch.cat(parallel_out, 1)

        if self.n_stages-self.parallel_stages > 0:
            for i in range(self.n_stages-self.parallel_stages):
                skip_list.append(x)
                x = self.down_sample[i](x)

        x = self.trans_block[0](x)

        for i in range(self.n_stages):
            skip_x = skip_list.pop()
            x = self.up_sample[i](x, skip_x)

        return x


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
            ailoc.transloc.Conv2d(c_input, c_output, 3, 1, 3//2),
            ailoc.transloc.Conv2d_GELU(c_output, c_output, 3, 1, 3//2))

        #  down sampling blocks
        curr_c = c_output
        self.down_sample = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_sample.append(ailoc.transloc.DownSampleBlock(curr_c, curr_c * 2, kernel_size))
            curr_c *= 2

        # up sampling blocks
        self.up_sample = nn.ModuleList()
        for i in range(n_stages):
            # add a heavy tail to the first up sampling block
            tail = None
            if i == 0:
                tail = []
                for j in range(3):
                    tail.append(nn.Sequential(
                        ailoc.transloc.ConvNextBlock(curr_c//2, curr_c//2, kernel_size),
                        ailoc.transloc.ConvNextBlock(curr_c//2, curr_c//2, kernel_size),
                    ))
                tail = nn.Sequential(*tail)
            self.up_sample.append(ailoc.transloc.UpSampleBlock(curr_c, curr_c // 2, kernel_size, tail))
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


class FAMUnet(nn.Module):
    """
    used for frame analysis module and temporal context module
    """

    def __init__(self, n_input, n_filters=48, n_stages=2, pad=1, ker_size=3):
        super().__init__()

        curr_n = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()

        self.layer_path.append(
            nn.Conv2d(in_channels=n_input, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_n *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            # self.layer_path.append(nn.UpsamplingBilinear2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_n //= 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n * 2, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_n, out_channels=curr_n, kernel_size=ker_size, padding=pad).cuda())

        for m in self.layer_path:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        self.transformer = ailoc.transloc.TemporalTransformer(seq_length=96,  # CNN feature channels in
                                                              embedding_dim=1024,  # CNN feature flatten size 32*32
                                                              num_layers=2,
                                                              num_heads=8,
                                                              hidden_dim=1024 * 4,  # embedding_dim * 4
                                                              input_dropout_rate=0,
                                                              attn_dropout_rate=0.1, )

    def forward(self, x):

        n_l = 0
        x_bridged = []

        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1
        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        x = self.transformer(x)

        for i in range(self.n_stages):
            for n in range(4):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1)  # concatenate

        return x


class TransLocNet(nn.Module):
    """
    The TransLocNet consists of a U-NeXt based feature extraction module(FEM),
    an optional temporal Transformer module(TTM) and an output head. The output representation is inspired by
    the DECODE.
    """

    def __init__(self, temporal_attn=True, attn_length=3, train_context_size=12):
        super().__init__()

        self.temporal_attn = temporal_attn
        self.attn_length = attn_length
        self.train_context_size = train_context_size
        self.n_features = 48

        self.fem = U_NeXt(c_input=1,
                          c_output=48,
                          n_stages=2,
                          kernel_size=5).cuda()
        if self.temporal_attn:
            # temporal transformer module
            patch_size = 1
            self.ttm = ailoc.transloc.TransformerBlock(seq_length=train_context_size,
                                                       attn_length=attn_length,
                                                       c_input=self.n_features,
                                                       patch_size=patch_size,
                                                       embedding_dim=(patch_size**2)*self.n_features,
                                                       num_layers=2,
                                                       num_heads=8,
                                                       mlp_dim=(patch_size**2)*4*self.n_features,
                                                       dropout_rate=0.0,
                                                       context_dropout=0.0).cuda()

            # # Unext based temporal context module
            # self.ttm = U_NeXt(c_input=self.n_features,
            #                   c_output=48,
            #                   n_stages=2,
            #                   kernel_size=5).cuda()

        self.om = Out_Head(c_input=48, kernel_size=3).cuda()

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

        # during training and online evaluation, the inpout should have the shape
        # (training batch size, context_size, height, width)
        if x_input.ndimension() == 4:
            batch_size, context_size = x_input.shape[:2]
            fem_out = self.fem(x_input.reshape([-1, 1, img_h, img_w]))
            if self.temporal_attn:
                # transformer based ttm
                fem_out = fem_out.reshape([batch_size, context_size, self.n_features, img_h, img_w])

                # # Unext based ttm
                # fem_out = fem_out.reshape([batch_size, context_size, -1, img_h, img_w])
                # zeros = torch.zeros_like(fem_out[:, :1])
                # h_t1 = fem_out
                # h_t0 = torch.cat([zeros, fem_out[:, :-1]], dim=1)
                # h_t2 = torch.cat([fem_out[:, 1:], zeros], dim=1)
                # fem_out = torch.cat([h_t0, h_t1, h_t2], dim=2)[:, 1:-1].reshape(-1, self.n_features, img_h, img_w)

        # when analyzing experimental data, the input dimension is 3, (anlz_batch_size, height, width)
        elif x_input.ndimension() == 3:
            anlz_batch_size = x_input.shape[0]
            fem_out = self.fem(x_input[:, None])
            if self.temporal_attn:
                # transformer based ttm
                fem_out = fem_out.reshape([1, anlz_batch_size, self.n_features, img_h, img_w])

                # # Unext based ttm
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

        if self.temporal_attn:
            # ttm_out = self.ttm(fem_out).reshape([-1, 48, img_h, img_w])
            ttm_out = self.ttm(fem_out)[self.attn_length//2: -(self.attn_length//2)]

            p, xyzph, xyzphs, bg = self.om(ttm_out)
        else:
            p, xyzph, xyzphs, bg = self.om(fem_out)

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

        print('Params:', params)
        print(f'MACs:{macs}, (input shape: {dummy_input.shape})')
