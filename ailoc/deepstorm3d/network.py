import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import time


# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


# Localization architecture
class LocalizationCNN(nn.Module):
    def __init__(self, local_context, attn_length, context_size, discret_z, scaling_factor):
        super(LocalizationCNN, self).__init__()

        self.local_context = local_context
        self.attn_length = attn_length
        self.train_context_size = context_size
        self.discret_z = discret_z
        self.scaling_factor = scaling_factor

        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        if True:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)
        else:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, self.discret_z, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(self.discret_z, self.discret_z, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(self.discret_z, self.discret_z, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(self.discret_z, self.discret_z, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=self.scaling_factor)  # hardtanh may die in some situations

    def forward(self, im):
        h,w = im.shape[-2:]
        im = im.reshape(-1,1,h,w)

        # extract multi-scale features
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = F.interpolate(features, scale_factor=2)
        out = self.deconv1(out)
        out = F.interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)

        return out

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
