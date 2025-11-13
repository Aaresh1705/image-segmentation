import torch.nn.functional as F
from torch import nn


class EncDecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'EncDecNet'

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
        self.upsample0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        p0 = self.pool0(e0)

        e1 = F.relu(self.enc_conv1(p0))
        p1 = self.pool1(e1)

        e2 = F.relu(self.enc_conv2(p1))
        p2 = self.pool2(e2)

        e3 = F.relu(self.enc_conv3(p2))
        p3 = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(p3))

        # decoder
        d0 = self.upsample0(b)
        d0 = F.relu(self.dec_conv1(d0))

        d1 = self.upsample1(d0)
        d1 = F.relu(self.dec_conv2(d1))

        d2 = self.upsample2(d1)
        d2 = F.relu(self.dec_conv3(d2))

        d3 = self.upsample3(d2)  # no activation
        out = self.dec_conv3(d3)
        return F.sigmoid(out)
