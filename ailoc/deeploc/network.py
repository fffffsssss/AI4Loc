import torch
import torch.nn as nn
import torch.nn.functional as F
import thop


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


class DeepLocNet(nn.Module):
    """
    DeepLocNet model, refer to DECODE network
    1. Speiser, A. et al. Deep learning enables fast and dense single-molecule localization
        with high accuracy. Nat Methods 18, 1082–1090 (2021).
    """

    def __init__(self, local_context=True):
        super().__init__()

        self.local_context = local_context
        self.n_features = 48 * 3 if self.local_context else 48

        self.frame_anlz_module = Unet(n_inp=1, n_filters=48, n_stages=2, pad=1, ker_size=3).cuda()
        self.temp_context_module = Unet(n_inp=self.n_features, n_filters=48, n_stages=2, pad=1, ker_size=3).cuda()
        self.out_module = Outnet(n_inp=48, pad=1, ker_size=3).cuda()

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
                fm_out = fm_out.reshape([batch_size, context_size, -1, img_h, img_w])
                zeros = torch.zeros_like(fm_out[:, :1])
                h_t1 = fm_out
                h_t0 = torch.cat([zeros, fm_out[:, :-1]], dim=1)
                h_t2 = torch.cat([fm_out[:, 1:], zeros], dim=1)
                fm_out = torch.cat([h_t0, h_t1, h_t2], dim=2)[:, 1:-1].reshape(-1, self.n_features, img_h, img_w)

        # when analyzing experimental data, the input dimension is 3, (analysis batch size, height, width)
        elif x_input.ndimension() == 3:
            x_input = x_input[:, None]
            fm_out = self.frame_anlz_module(x_input)  # (analysis batch size, 48, height, width)

            # if using local context, the ailoc.common.analyze.data_analyze() function will apply
            # the rolling inference strategy, the input will be padded with two neighbouring frames at
            # the head and tail to provide local context for target frames, so after context feature
            # concatenation, the head and tail output in the batch could be discarded
            if self.local_context:
                # create a zero output with the shape (1, 48, height, width) to pad the head and tail
                zeros = torch.zeros_like(fm_out[:1])
                # the fm_out has the shape (analysis batch size+2, 48, height, width)
                h_t1 = fm_out
                # pad zero to the head and discard the tail frame output of the batch
                h_t0 = torch.cat([zeros, fm_out], 0)[:-1]
                # pad zero to the tail and discard the head frame output of the batch
                h_t2 = torch.cat([fm_out, zeros], 0)[1:]
                # reuse the features of each frame to build the local context with the shape
                # (analysis batch size+2, 144, height, width), discard the head and tail
                fm_out = torch.cat([h_t0, h_t1, h_t2], 1)[1:-1]

        else:
            raise ValueError('The input dimension is not supported.')

        # layer normalization
        fm_out_ln = nn.functional.layer_norm(fm_out, normalized_shape=[self.n_features, img_h, img_w])

        cm_out = self.temp_context_module(fm_out_ln)
        p, xyzph, xyzphs, bg = self.out_module(cm_out)
        # p, xyzph, xyzphs, bg = self.out_module(fm_out_ln)  # skip the temporal context module

        # todo:  may need to set an extra scale and offset for the output considering the sensitive
        #  range of the activation function sigmoid and tanh
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
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(1, 12, 64, 64).cuda() if self.local_context else torch.randn(1, 10, 64, 64).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')

        print('Params:', params)
        print(f'MACs:{macs}, (input shape: {dummy_input.shape})')




