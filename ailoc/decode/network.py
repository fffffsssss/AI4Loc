import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import time


class Unet(nn.Module):
    """
    used for frame analysis module and temporal context module
    """

    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3):
        super().__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()

        self.layer_path.append(
            nn.Conv2d(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_N *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_N //= 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

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


class Outnet(nn.Module):
    """
    output module
    """

    def __init__(self, n_inp, pad=1, ker_size=3):
        super().__init__()

        self.p_out1 = nn.Conv2d(in_channels=n_inp, out_channels=n_inp, kernel_size=ker_size,
                                padding=pad).cuda()
        self.p_out2 = nn.Conv2d(in_channels=n_inp, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzph_out1 = nn.Conv2d(in_channels=n_inp, out_channels=n_inp, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.xyzph_out2 = nn.Conv2d(in_channels=n_inp, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.p_out2.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzph_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzph_out2.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.zeros_(self.xyzph_out2.bias)

        self.xyzphs_out1 = nn.Conv2d(in_channels=n_inp, out_channels=n_inp, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzphs_out2 = nn.Conv2d(in_channels=n_inp, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzphs_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzphs_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzphs_out2.bias)

        self.bg_out1 = nn.Conv2d(in_channels=n_inp, out_channels=n_inp, kernel_size=ker_size,
                                 padding=pad).cuda()
        self.bg_out2 = nn.Conv2d(in_channels=n_inp, out_channels=1, kernel_size=1, padding=0).cuda()

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


class DECODENet(nn.Module):
    """
    DECODE network
    1. Speiser, A. et al. Deep learning enables fast and dense single-molecule localization
        with high accuracy. Nat Methods 18, 1082–1090 (2021).
    """

    def __init__(self, local_context=True, attn_length=3, train_context_size=12):
        super().__init__()

        self.local_context = local_context
        self.attn_length = attn_length
        self.train_context_size = train_context_size
        self.n_features = 48

        self.frame_anlz_module = Unet(n_inp=1,
                                      n_filters=self.n_features,
                                      n_stages=2,
                                      pad=1,
                                      ker_size=3).cuda()
        self.temp_context_module = Unet(n_inp=self.n_features*self.attn_length if self.local_context else self.n_features,
                                        n_filters=self.n_features,
                                        n_stages=2,
                                        pad=1,
                                        ker_size=3).cuda()
        self.out_module = Outnet(n_inp=self.n_features,
                                 pad=1,
                                 ker_size=3).cuda()

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
            fm_out = self.frame_anlz_module(x_input.reshape([-1, 1, img_h, img_w]))

            if self.local_context:
                extra_length = self.attn_length // 2
                fm_out = fm_out.reshape([batch_size, context_size, -1, img_h, img_w])
                temporal_list = []
                for i in range(extra_length):
                    zeros = torch.zeros_like(fm_out[:, :(extra_length-i)])
                    temporal_list.append(torch.cat([zeros, fm_out[:,:-(extra_length-i)]], dim=1))
                temporal_list.append(fm_out)
                for i in range(extra_length):
                    zeros = torch.zeros_like(fm_out[:, :i+1])
                    temporal_list.append(torch.cat([fm_out[:, i+1:], zeros], dim=1))
                fm_out = (torch.cat(temporal_list, dim=2)[:, extra_length:-extra_length]
                          .reshape(-1, self.n_features*self.attn_length, img_h, img_w))

        # when analyzing experimental data, the input dimension is 3, (analysis batch size, height, width)
        elif x_input.ndimension() == 3:
            x_input = x_input[:, None]
            fm_out = self.frame_anlz_module(x_input)  # (analysis batch size, 48, height, width)

            # if using local context, the ailoc.common.analyze.data_analyze() function will apply
            # the rolling inference strategy, the input will be padded with two neighbouring frames at
            # the head and tail to provide local context for target frames, so after context feature
            # concatenation, the head and tail output in the batch could be discarded
            if self.local_context:
                extra_length = self.attn_length // 2
                temporal_list = []
                for i in range(extra_length):
                    zeros = torch.zeros_like(fm_out[:(extra_length-i)])
                    temporal_list.append(torch.cat([zeros, fm_out[:-(extra_length - i)]], dim=0))
                temporal_list.append(fm_out)
                for i in range(extra_length):
                    zeros = torch.zeros_like(fm_out[:i + 1])
                    temporal_list.append(torch.cat([fm_out[i + 1:], zeros], dim=0))
                fm_out = torch.cat(temporal_list, dim=1)[extra_length:-extra_length]
        else:
            raise ValueError('The input dimension is not supported.')

        cm_in = fm_out

        cm_out = self.temp_context_module(cm_in)
        p, xyzph, xyzphs, bg = self.out_module(cm_out)
        # p, xyzph, xyzphs, bg = self.out_module(cm_in)  # skip the temporal context module

        # p_pred = torch.sigmoid(torch.clamp(outputs['p'], min=-16, max=16))[:, 0]  # probability
        p_pred = torch.sigmoid(torch.clamp(p, min=-16, max=16))[:, 0]  # probability

        # xyzph_pred = outputs['xyzph']
        xyzph_pred = xyzph

        # output xy range is (-1, 1), the training sampling range is (-0.5,0.5)
        xyzph_pred[:, :2] = torch.tanh(xyzph_pred[:, :2])

        # output z range is (-2, 2), the training sampling range is in (-1,1)
        xyzph_pred[:, 2] = torch.tanh(xyzph_pred[:, 2]) * 2

        # output photon range is (0, 1), the training sampling range is in (0,1)
        xyzph_pred[:, 3] = torch.sigmoid(xyzph_pred[:, 3])

        # scale the uncertainty and add epsilon, the output range becomes (0.0001, 3.0001),
        # maybe can use RELU for unlimited upper range and stable gradient
        # xyzph_sig_pred = torch.sigmoid(outputs['xyzph_sig']) * 3 + 0.0001
        xyzph_sig_pred = torch.sigmoid(xyzphs) * 3 + 0.0001

        # output bg range is (0, 1), the training sampling range is in (0,1)
        # bg_pred = torch.sigmoid(outputs['bg'][:, 0])  # bg
        bg_pred = torch.sigmoid(bg[:, 0])  # bg

        return p_pred, xyzph_pred, xyzph_sig_pred, bg_pred

    def get_parameter_number(self):
        print('-' * 200)
        print('Testing network parameters and multiply-accumulate operations (MACs)')
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(1, self.train_context_size, 64, 64).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(200):
            self.forward(dummy_input)
        print(f'Average forward time: {(time.time() - t0) / 200:.4f} s')
