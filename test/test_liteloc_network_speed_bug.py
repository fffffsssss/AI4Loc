import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import time

torch.backends.cudnn.benchmark = True


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
        self.n_features = 48

        self.frame_anlz_module = Unet(n_inp=1,
                                      n_filters=self.n_features,
                                      n_stages=2,
                                      pad=1,
                                      ker_size=3).cuda()
        self.temp_context_module = Unet(n_inp=self.n_features*3 if self.local_context else self.n_features,
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
                fm_out = fm_out.reshape([batch_size, context_size, -1, img_h, img_w])
                zeros = torch.zeros_like(fm_out[:, :1])
                h_t1 = fm_out
                h_t0 = torch.cat([zeros, fm_out[:, :-1]], dim=1)
                h_t2 = torch.cat([fm_out[:, 1:], zeros], dim=1)
                fm_out = torch.cat([h_t0, h_t1, h_t2], dim=2)[:, 1:-1].reshape(-1, self.n_features*3, img_h, img_w)

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
        fm_out_ln = nn.functional.layer_norm(fm_out, normalized_shape=fm_out.shape[1:])

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
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(200):
            self.forward(dummy_input)
        print(f'Average forward time: {(time.time() - t0) / 200:.4f} s')


class OutnetCoordConv(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutnetCoordConv, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)



    def forward(self, x):

        outputs = {}
        p = F.elu(self.p_out(x))
        outputs['p'] = self.p_out1(p)

        xyzi =F.elu(self.xyzi_out(x))
        outputs['xyzi'] = self.xyzi_out1(xyzi)
        xyzis = F.elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
        return outputs


# Define the basic Conv-LeakyReLU-BN
class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.lrelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


class LiteLocNet(nn.Module):
    def __init__(self, dilation_flag):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True).cuda()
        self.layer1 = Conv2DReLUBN(1, 64, 3, 1, 1).cuda()  # replace Conv2d
        self.layer2 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1).cuda()
        self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1).cuda()
        if dilation_flag:
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, (2, 2), (2, 2)).cuda() # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, (4, 4), (4, 4)).cuda()  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, (8, 8), (8, 8)).cuda()
            self.layer7 = Conv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16)).cuda()
            self.layer71 = Conv2DReLUBN(64+ 1 , 64, 3, (16, 16), (16, 16)).cuda()
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1).cuda()
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1).cuda()
        self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True).cuda()
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool = nn.AvgPool2d(2, stride=2).cuda()
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        # self.pred = OutnetCoordConv(64, 1, 3)

        self.out_module = Outnet(n_inp=64,
                                 pad=1,
                                 ker_size=3).cuda()

        self.local_context = False
        self.get_parameter_number()

    def forward(self, im):

        img_h, img_w = im.shape[-2], im.shape[-1]
        im = im.view(-1, 1, img_h, img_w)

        # extract multi-scale features
        out = self.norm(im)

        out = self.layer1(out)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)

        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out4 = self.layer4(features) + out

        features = torch.cat((out4, im), 1)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)
        out = self.layer7(features) + out4 + out6
        features = torch.cat((out, im), 1)

        out1 = self.deconv1(features)
        out = self.pool(out1)
        out = self.layerU1(out)
        out = self.layerU2(out)
        out = self.layerU3(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.layerD3(out)
        out = torch.cat([out, out1], 1)
        out = self.layerD2(out)
        out = self.layerD1(out)

        p, xyzph, xyzphs, bg = self.out_module(out)
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
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(200):
            self.forward(dummy_input)
        print(f'Average forward time: {(time.time() - t0) / 200:.4f} s')


if __name__ == '__main__':
    model1 = LiteLocNet(dilation_flag=True)
    model2 = DeepLocNet(local_context=False)
