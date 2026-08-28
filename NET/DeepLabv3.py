# DeepLabv3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        rate1, rate2, rate3 = atrous_rates

        # rate = 1 分支（1x1 conv）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # rate = 6, 12, 18
        self.branch2 = ASPPConv(in_channels, out_channels, rate1)
        self.branch3 = ASPPConv(in_channels, out_channels, rate2)
        self.branch4 = ASPPConv(in_channels, out_channels, rate3)

        # image-level pooling
        self.branch5 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x1 = self.branch1(x)  # rate=1
        x2 = self.branch2(x)  # rate=6
        x3 = self.branch3(x)  # rate=12
        x4 = self.branch4(x)  # rate=18
        x5 = self.branch5(x)  # image pooling

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x


# --------------------------------
# DeepLabv3 head
# --------------------------------

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.aspp = ASPP(in_channels, 256, atrous_rates)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(
        self,
        num_classes=21,
        backbone="resnet50",
        pretrained=False,
    ):
        super().__init__()

        # 原始 DeepLabv3 常用 output_stride = 16
        replace_stride_with_dilation = [False, False, True]  # 只有 layer4 用空洞
        atrous_rates = (6, 12, 18)

        # ResNet backbone（帶空洞卷積設定）
        if backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            out_channels = 2048
        elif backbone == "resnet101":
            resnet = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            out_channels = 2048
        else:
            raise ValueError("backbone 只能是 'resnet50' 或 'resnet101'")

        # conv1 ~ layer4 當 feature extractor
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.classifier = DeepLabHead(out_channels, num_classes, atrous_rates)

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.backbone(x)   # [B, 2048, H/16, W/16]
        x = self.classifier(features) # [B, C, H/16, W/16]

        x = F.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )
        return x


# --------------------------------
# Example
# --------------------------------
if __name__ == "__main__":
    model = DeepLabV3(num_classes=2,backbone="resnet50",pretrained=False)
    img = torch.randn(4, 3, 256, 256)
    out = model(img)
    print(out.shape)  # torch.Size([4, 2, 256, 256])