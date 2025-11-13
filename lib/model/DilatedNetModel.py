import torch.nn.functional as F
from torch import nn


class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'DilatedNet'

        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2)
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, dilation=4, padding=4)
        self.enc_conv3 = nn.Conv2d(64, 64, kernel_size=3, dilation=8, padding=8)

        self.bottleneck_conv = nn.Conv2d(64, 64, kernel_size=3, dilation=16, padding=16)

        # No upsampling necessary (resolution never changed)
        self.dec_conv0 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e0 = F.relu(self.enc_conv0(x))  # dilation 1
        e1 = F.relu(self.enc_conv1(e0)) # dilation 2
        e2 = F.relu(self.enc_conv2(e1)) # dilation 4
        e3 = F.relu(self.enc_conv3(e2)) # dilation 8

        b = F.relu(self.bottleneck_conv(e3))  # dilation 16

        d0 = F.relu(self.dec_conv0(b))
        d1 = F.relu(self.dec_conv1(d0))
        d2 = F.relu(self.dec_conv2(d1))
        d3 = F.relu(self.dec_conv3(d2))

        # ----- Output -----
        out = self.out_conv(d3)
        return F.sigmoid(out)
