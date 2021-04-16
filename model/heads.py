import torch.nn as nn
import torch
import torch.nn.functional as F


class ChangeHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=2):
        super(ChangeHead, self).__init__()
        inter_channel = in_channels

        self.sam = nn.Sequential(nn.Conv2d(in_channels, inter_channel, kernel_size=3, padding=1, stride=1, bias=False),
                                 nn.BatchNorm2d(inter_channel),
                                 nn.ReLU(inplace=True),
                                 )

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.seg_conv = nn.Conv2d(inter_channel, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        out = x
        out = self.sam(out)
        out = self.upsampling(out)
        out = self.seg_conv(out)
        out = torch.sigmoid(out)
        return out
