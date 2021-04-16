import torch
import torch.nn as nn
import torch.nn.functional as F

Activation = nn.ReLU


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        self.upsample = n_upsamples

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=int(
                2 ** self.upsample), mode="bilinear", align_corners=True)
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    self.policy)
            )


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm=True, activation=False):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.norm = norm
        self.activation = activation
        bias = True
        if self.norm:
            self.bn = nn.BatchNorm2d(out_channels)
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=bias)
        if self.activation:
            self.activation = Activation()

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.bn(out)
        if self.activation:
            out = self.activation(out)
        return out


class FPM(nn.Module):

    def __init__(self, encoder_channels, pyramid_channel, encoder_size):
        super(FPM, self).__init__()

        self.epsilon = 1e-4

        # for UP
        self.c1_down_channel = ConvBlock(
            encoder_channels[0], pyramid_channel, kernel_size=1, activation=True)
        self.c2_down_channel = ConvBlock(
            encoder_channels[1], pyramid_channel, kernel_size=1, activation=True)
        self.c3_down_channel = ConvBlock(
            encoder_channels[2], pyramid_channel, kernel_size=1, activation=True)
        self.c4_down_channel = ConvBlock(
            encoder_channels[3], pyramid_channel, kernel_size=1, activation=True)
        self.c5_down_channel = ConvBlock(
            encoder_channels[4], pyramid_channel, kernel_size=1, activation=True)

        # up path

        # # mid fusion
        self.n5_conv_up = ConvBlock(
            pyramid_channel, pyramid_channel, kernel_size=3, activation=True)
        self.n4_conv_up = ConvBlock(
            pyramid_channel, pyramid_channel, kernel_size=3, activation=True)
        self.n3_conv_up = ConvBlock(
            pyramid_channel, pyramid_channel, kernel_size=3, activation=True)
        self.n2_conv_up = ConvBlock(
            pyramid_channel, pyramid_channel, kernel_size=3, activation=True)
        self.n1_conv_up = ConvBlock(
            pyramid_channel, pyramid_channel, kernel_size=3, activation=True)

        # # up sample
        self.n5_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.n4_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.n3_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.n2_upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, *features):
        c1, c2, c3, c4, c5 = features[:]

        # up path
        c1_in = self.c1_down_channel(c1)
        c2_in = self.c2_down_channel(c2)
        c3_in = self.c3_down_channel(c3)
        c4_in = self.c4_down_channel(c4)
        c5_in = self.c5_down_channel(c5)

        n5 = c5_in

        n4 = self.n5_upsample(n5) + c4_in

        n3 = self.n4_upsample(n4) + c3_in

        n2 = self.n3_upsample(n3) + c2_in

        n1 = self.n2_upsample(n2) + c1_in

        p1 = self.n1_conv_up(n1)

        p2 = self.n2_conv_up(n2)

        p3 = self.n3_conv_up(n3)

        p4 = self.n4_conv_up(n4)

        p5 = self.n5_conv_up(n5)

        return p1, p2, p3, p4, p5


class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            change_channels=64,
            pyramid_channel=128,
            merge_policy="add",
            encoder_size=None,
    ):
        super().__init__()

        if encoder_size is None:
            encoder_size = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]

        self.out_channels = pyramid_channel if merge_policy == "add" else pyramid_channel * 5

        encoder_channels = encoder_channels[1:]

        self.dff1 = FPM(encoder_channels, pyramid_channel, encoder_size)

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channel, change_channels, n_upsamples)
            for n_upsamples in [0, 1, 2, 3, 4]
        ])
        self.merge = MergeBlock(merge_policy)

    def forward(self, features1, features2):
        features1 = features1[1:]
        features2 = features2[1:]
        c1, c2, c3, c4, c5 = [torch.abs(features1[0] - features2[0]),
                              torch.abs(features1[1] - features2[1]),
                              torch.abs(features1[2] - features2[2]),
                              torch.abs(features1[3] - features2[3]),
                              torch.abs(features1[4] - features2[4]),
                              ]

        p1, p2, p3, p4, p5 = self.dff1(c1, c2, c3, c4, c5)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p1, p2, p3, p4, p5])]
        x = self.merge(feature_pyramid)

        return x
