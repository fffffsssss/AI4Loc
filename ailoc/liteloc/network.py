import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import time


class OutNet(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutNet, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad)

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad)
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0) # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0)  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad)
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0)

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)

    def forward(self, x):

        outputs = {}
        p = F.elu(self.p_out(x))
        outputs['p'] = self.p_out1(p)

        xyzi = F.elu(self.xyzi_out(x))
        outputs['xyzi'] = self.xyzi_out1(xyzi)
        xyzis = F.elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
        return outputs


class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.lrelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)
        # self.fuse = FusedResNetBlock(self.conv, self.bn).cuda()

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)  # revert the order of bn and relu
        return out


class LiteLocNet(nn.Module):
    def __init__(self, local_context=True, attn_length=3, train_context_size=12):
        super().__init__()

        self.local_context = local_context
        if not self.local_context:
            attn_length = 1
        self.attn_length = attn_length
        self.train_context_size = train_context_size
        self.n_features = 64

        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer0 = Conv2DReLUBN(1, self.n_features, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(self.n_features, self.n_features, 3, 1, 1)
        self.layer2 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(self.n_features*(1+self.attn_length), self.n_features, 3, 1, 1)

        self.layer4 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
        self.layer5 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
        self.layer6 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, (8, 8), (8, 8))
        self.layer7 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, (16, 16), (16, 16))

        self.deconv1 = Conv2DReLUBN(self.n_features*2, self.n_features, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(self.n_features, self.n_features * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(self.n_features * 2, self.n_features * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(self.n_features * 2, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(self.n_features * 2, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.pred = OutNet(self.n_features, 1, 3)

    def forward(self, x_input):
        img_h, img_w = x_input.shape[-2], x_input.shape[-1]
        if x_input.ndimension() == 4:
            batch_size, context_size = x_input.shape[:2]

        x0 = x_input.reshape([-1, 1, img_h, img_w])

        if self.local_context:
            extra_length = self.attn_length // 2

        # down-sampling layer
        x1 = self.norm(x0)
        x2 = self.layer0(x1)
        x3 = self.pool1(x2)

        # prepare the skip concatenation
        if x_input.ndimension() == 4:
            x3 = x3.reshape([batch_size, context_size, -1, x3.shape[-2], x3.shape[-1]])
            if self.local_context:
                x_cat = x3[:, extra_length:-extra_length].reshape(-1, self.n_features, x3.shape[-2], x3.shape[-1])
            else:
                x_cat = x3.reshape(-1, self.n_features, x3.shape[-2], x3.shape[-1])
            x3 = x3.reshape(-1, self.n_features, x3.shape[-2], x3.shape[-1])
        elif x_input.ndimension() == 3:
            if self.local_context:
                x_cat = x3[extra_length:-extra_length].reshape(-1, self.n_features, x3.shape[-2], x3.shape[-1])
            else:
                x_cat = x3.reshape(-1, self.n_features, x3.shape[-2], x3.shape[-1])

        x4 = self.layer1(x3)  # (12, 64, 64, 64)
        x5 = torch.cat((x4, x3), 1)  # (12, 128, 64, 64)
        x6 = self.layer2(x5) + x4  # (12, 64, 64, 64)
        x7 = torch.cat((x6, x3), 1)  # (12, 128, 64, 64)
        x8 = self.layer3(x7) + x6  # (12, 64, 64, 64)

        # during training and online evaluation, the x_inpout should have the shape
        # (training batch size, context_size, height, width)
        if x_input.ndimension() == 4:
            if self.local_context:
                x8 = x8.reshape([batch_size, context_size, -1, x8.shape[-2], x8.shape[-1]])
                temporal_list = []
                for i in range(extra_length):
                    zeros = torch.zeros_like(x8[:, :(extra_length-i)])
                    temporal_list.append(torch.cat([zeros, x8[:,:-(extra_length-i)]], dim=1))
                temporal_list.append(x8)
                for i in range(extra_length):
                    zeros = torch.zeros_like(x8[:, :i+1])
                    temporal_list.append(torch.cat([x8[:, i+1:], zeros], dim=1))
                out = (torch.cat(temporal_list, dim=2)[:, extra_length:-extra_length]
                       .reshape(-1, self.n_features*self.attn_length, x8.shape[-2], x8.shape[-1]))
                features = torch.cat((out, x_cat), 1)
                x8 = self.layer30(features) + x8[:,extra_length:-extra_length].reshape([-1,
                                                                                         self.n_features,
                                                                                         x8.shape[-2],
                                                                                         x8.shape[-1]])
            else:
                features = torch.cat((x8, x_cat), 1)
                x8 = self.layer30(features) + x8

        # when analyzing experimental data, the input dimension is 3, (analysis batch size, height, width)
        elif x_input.ndimension() == 3:
            if self.local_context:
                temporal_list = []
                for i in range(extra_length):
                    zeros = torch.zeros_like(x8[:(extra_length - i)])
                    temporal_list.append(torch.cat([zeros, x8[:-(extra_length - i)]], dim=0))
                temporal_list.append(x8)
                for i in range(extra_length):
                    zeros = torch.zeros_like(x8[:i + 1])
                    temporal_list.append(torch.cat([x8[i + 1:], zeros], dim=0))
                out = torch.cat(temporal_list, dim=1)[extra_length:-extra_length]
                features = torch.cat((out, x_cat), 1)
                x8 = self.layer30(features) + x8[extra_length:-extra_length]
            else:
                features = torch.cat((x8, x_cat), 1)
                x8 = self.layer30(features) + x8

        # shallow feature extractor
        x9 = torch.cat((x8, x_cat), 1)
        x10 = self.layer4(x9) + x8
        x11 = torch.cat((x10, x_cat), 1)  # (10, 128, 64, 64)
        x12 = self.layer5(x11) + x10  # (10, 64, 64, 64)
        x13 = torch.cat((x12, x_cat), 1)  # (10, 128, 64, 64)
        x14 = self.layer6(x13) + x12  # (10, 64, 64, 64)
        x15 = torch.cat((x14, x_cat), 1)  # (10, 128, 64, 64)
        x16 = self.layer7(x15) + x14 + x10  # (10, 64, 64, 64)
        x17 = torch.cat((x16, x_cat), 1)  # (10, 128, 64, 64)

        out1 = self.deconv1(x17)  # (10, 64, 64, 64)

        # deep feature extractor
        out = self.pool(out1)  # (10, 64, 32, 32)
        out = self.layerU1(out)  # (10, 64, 32, 32)
        out = self.layerU2(out)  # (10, 128, 32, 32)
        out = self.layerU3(out)  # (10, 128, 32, 32)
        out = F.interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
        out = self.layerD3(out)  # (10, 64, 64, 64)
        out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
        out = self.layerD2(out)  # (10, 64, 64, 64)
        out = self.layerD1(out)  # (10, 64, 64, 64)
        out = F.interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
        out = self.layerD0(out)  # (10, 64, 128, 128)
        out = self.layerD00(out)  # (10, 64, 128, 128)

        # output module
        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001

        bg_pred =None

        return probs[:, 0], xyzi_est, xyzi_sig, bg_pred

    def post_process(self, p, xyzi_est):

        xyzi_est = xyzi_est.to(torch.float32)

        p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]

        # localize maximum values within a 3x3 patch
        pool = F.max_pool2d(p_clip, 3, 1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        # Add probability values from the 4 adjacent pixels
        filt = torch.Tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]).half().cuda()  # maybe half tensor affect the precision of result
        conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1, bias=None)
        p_ps1 = max_mask1 * conv

        # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask
        p_copy = p * (1 - max_mask1[:, 0])

        # p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
        max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]  # fushuang
        p_ps2 = max_mask2 * conv

        p = p_ps1 + p_ps2
        p = p[:, 0]

        xyzi_est[:, 0] += 0.5
        xyzi_est[:, 1] += 0.5

        p_index = torch.where(p > 0.3)
        frame_index = torch.unsqueeze(p_index[0], dim=1) + 1

        x = ((xyzi_est[:, 0])[p_index] + p_index[2]).unsqueeze(1)
        y = ((xyzi_est[:, 1])[p_index] + p_index[1]).unsqueeze(1)

        z = ((xyzi_est[:, 2])[p_index]).unsqueeze(1)
        ints = ((xyzi_est[:, 3])[p_index]).unsqueeze(1)
        p = (p[p_index]).unsqueeze(1)

        molecule_array = torch.cat([frame_index, x, y, z, ints, p], dim=1)

        return molecule_array

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

