# DeepLabv3plus加入cbam
import torch
import torch.nn as nn
import torch.nn.functional as F


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

# ------------------------
# Bottleneck Block for ResNet-50 (with dilation)  # CHANGED
# ------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, dilation=1, downsample=None):  # CHANGED
        super(Bottleneck, self).__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 conv 使用 dilation / padding 對應
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=dilation, dilation=dilation, bias=False)  # CHANGED
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ------------------------
# ResNet-50-like Backbone (輸出通道回到 2048，高階空洞保持 OS=16)  # CHANGED
# ------------------------
class HorizontalResNetBackbone(nn.Module):
    def __init__(self, output_stride=16):  # 可切換 OS=16 / OS=8
        super(HorizontalResNetBackbone, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #ResNet-50 配置：layer1(3), layer2(4), layer3(6), layer4(3)
        #out_channels = mid*4 : 64->256, 128->512, 256->1024, 512->2048
        #OS=16: layer3 stride=2, layer4 stride=1 + dil=2
        #OS=8 : layer3 stride=1 + dil=2, layer4 stride=1 + dil=4
        if output_stride == 16:
            dil3, str3 = 1, 2
            dil4, str4 = 2, 1
        else:  # OS=8
            dil3, str3 = 2, 1
            dil4, str4 = 4, 1

        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1, dilation=1)     # out 256
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2, dilation=1)     # out 512
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=str3, dilation=dil3)  # out 1024  # CHANGED
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=str4, dilation=dil4)  # out 2048  # CHANGED

    def _make_layer(self, in_channels, mid_channels, blocks, stride=1, dilation=1):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [Bottleneck(in_channels, mid_channels, stride=stride, dilation=dilation, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        low_level_feat = self.layer1(x)   # 256-ch 低階特徵（與原始 DeepLabV3+ 對齊）
        x = self.layer2(low_level_feat)   # 512
        x = self.layer3(x)                # 1024
        x = self.layer4(x)                # 2048（高階特徵）
        return x, low_level_feat

# ------------------------
# ASPP Module（輸入回 2048）  # CHANGED
# ------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, rates=(6, 12, 18)):  # CHANGED
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        feat5 = self.global_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)
        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(x)

# ------------------------
# Decoder（維持：低階 256 -> 48，與 ASPP 256 concat）
# ------------------------
class Decoder(nn.Module):
    #"""
    #DeepLabV3+ 論文原始版：
    #- 低階特徵 256 -> 48 (1x1)
    #- ASPP(256) 上採樣到與低階同尺寸
    #- Concat 後只接 **一層** 3x3 conv -> 256
    #"""
    def __init__(self, low_level_inplanes=256, low_level_outplanes=48, out_channels=256):
        super(Decoder, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_inplanes, low_level_outplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_outplanes),
            nn.ReLU(inplace=True)
        )
        # 論文版：concat 後只一層 3x3
        self.fuse = nn.Sequential(
            nn.Conv2d(low_level_outplanes + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_level_feat):
        low = self.conv_low(low_level_feat)                     # (48, H/4, W/4)
        x = F.interpolate(x, size=low.shape[2:], mode='bilinear', align_corners=True)  # (256, H/4, W/4)
        x = torch.cat([x, low], dim=1)                          # (256+48=304, H/4, W/4)
        x = self.fuse(x)                                        # (256, H/4, W/4)
        return x

# ------------------------
# DeepLabV3Plus（回到原始配置：ASPP in=2048, Decoder low=256→48, OS=16）
# ------------------------
class DeepLabV3Plus_CBAM(nn.Module):
    def __init__(self, num_classes=2, output_stride=16):
        super(DeepLabV3Plus_CBAM, self).__init__()
        self.backbone = HorizontalResNetBackbone(output_stride=output_stride)  # CHANGED
        self.aspp = ASPP(in_channels=2048, out_channels=256)                   # CHANGED
        self.decoder = Decoder(low_level_inplanes=256, low_level_outplanes=48, out_channels=256)
        self.cbam = CBAM(256) 
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        features, low_feat = self.backbone(x)
        x = self.aspp(features)
        x = self.decoder(x, low_feat)
        x = self.cbam(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    model = DeepLabV3Plus_CBAM(num_classes=2, output_stride=16)
    y = model(x)
    print(y.shape) 