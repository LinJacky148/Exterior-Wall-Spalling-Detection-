# Unet加入cbam
import torch
from torch import nn
from torch.nn import functional as F


# ------------------------- CBAM -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ----------------------------
# Basic Blocks
# ----------------------------
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


# ----------------------------
# UNet + Decoder CBAM
# ----------------------------
class UNet_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)

        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)

        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)

        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)

        # Bottleneck
        self.c5 = Conv_Block(512, 1024)

        # Decoder
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.cbam1 = CBAM(512)

        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.cbam2 = CBAM(256)

        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.cbam3 = CBAM(128)

        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.cbam4 = CBAM(64)

        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        # Encoder
        R1 = self.c1(x)     # 64
        D1 = self.d1(R1)

        R2 = self.c2(D1)    # 128
        D2 = self.d2(R2)

        R3 = self.c3(D2)    # 256
        D3 = self.d3(R3)

        R4 = self.c4(D3)    # 512
        D4 = self.d4(R4)

        # Bottleneck
        R5 = self.c5(D4)    # 1024

        # Decoder
        O1 = self.cbam1(self.c6(self.u1(R5, R4)))   # 512
        O2 = self.cbam2(self.c7(self.u2(O1, R3)))   # 256
        O3 = self.cbam3(self.c8(self.u3(O2, R2)))   # 128
        O4 = self.cbam4(self.c9(self.u4(O3, R1)))   # 64

        return self.out(O4)


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    net = UNet_CBAM(num_classes=2)
    y = net(x)
    print(y.shape)   # torch.Size([4, 2, 256, 256])