# PSPnet加入cbam
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn


# ------------------------- CBAM -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


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


# ------------------------- PPM -------------------------
class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        outputs = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(
                ppm(x),
                size=x.size()[-2:],
                mode='bilinear',
                align_corners=True
            )
            outputs.append(ppm_out)
        return outputs


class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=2):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)

        self.final = nn.Sequential(
            nn.Conv2d(
                self.in_channels + len(self.pool_sizes) * self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class Aux_Head(nn.Module):
    def __init__(self, in_channels=1024, num_classes=2):
        super(Aux_Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decode_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.in_channels // 2, self.in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.in_channels // 4, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decode_head(x)


# ------------------------- PSPNet + CBAM -------------------------
class Pspnet_CBAM(nn.Module):
    def __init__(self, num_classes, aux_loss=True):
        super(Pspnet_CBAM, self).__init__()
        self.num_classes = num_classes

        self.backbone = IntermediateLayerGetter(
            resnet50(weights=None, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer3': 'aux', 'layer4': 'stage4'}
        )

        self.aux_loss = aux_loss

        self.decoder = PSPHEAD(
            in_channels=2048,
            out_channels=512,
            pool_sizes=[1, 2, 3, 6],
            num_classes=self.num_classes
        )

        # CBAM after PPM output
        self.cbam = CBAM(512)

        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
        )

        if self.aux_loss:
            self.aux_head = Aux_Head(in_channels=1024, num_classes=self.num_classes)

    def forward(self, x):
        _, _, h, w = x.size()
        feats = self.backbone(x)

        x = self.decoder(feats["stage4"])   # PPM output
        x = self.cbam(x)                    # CBAM after PPM
        x = self.cls_seg(x)

        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    net = Pspnet_CBAM(num_classes=2)
    output = net(x)
    print(output.shape)