import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "UNet"

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64

        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16

        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        # use scale_factor=2 to be resolution-agnostic
        self.upsample0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 64 -> 128
        # note: last layer doesn't concat with encoder; stays at 64 channels
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))   # 3  -> 64, 128x128
        p0 = self.pool0(e0)              # 64 -> 64, 64x64

        e1 = F.relu(self.enc_conv1(p0))  # 64 -> 64, 64x64
        p1 = self.pool1(e1)              # 64 -> 64, 32x32

        e2 = F.relu(self.enc_conv2(p1))  # 64 -> 64, 32x32
        p2 = self.pool2(e2)              # 64 -> 64, 16x16

        e3 = F.relu(self.enc_conv3(p2))  # 64 -> 64, 16x16
        p3 = self.pool3(e3)              # 64 -> 64, 8x8

        # bottleneck
        b = F.relu(self.bottleneck_conv(p3))  # 64, 8x8

        # decoder
        d0 = self.upsample0(b)                               # 64, 16x16
        d0 = torch.cat([d0, e3], dim=1)                      # 128, 16x16
        d0 = F.relu(self.dec_conv0(d0))                      # 64, 16x16

        d1 = self.upsample1(d0)                              # 64, 32x32
        d1 = torch.cat([d1, e2], dim=1)                      # 128, 32x32
        d1 = F.relu(self.dec_conv1(d1))                      # 64, 32x32

        d2 = self.upsample2(d1)                              # 64, 64x64
        d2 = torch.cat([d2, e1], dim=1)                      # 128, 64x64
        d2 = F.relu(self.dec_conv2(d2))                      # 64, 64x64

        d3 = self.upsample3(d2)                              # 64, 128x128
        out = self.dec_conv3(d3)                             # 1, 128x128

        return torch.sigmoid(out)


# TODO : Implement a second homemade UNet
# class UNet2:
#     ....
#     ....
