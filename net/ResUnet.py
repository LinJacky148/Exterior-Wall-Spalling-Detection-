# ResUnet50 
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# -------------------- ResNet Utils --------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# -------------------- ResNet Blocks --------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -------------------- ResNet Backbone --------------------
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)
        x = self.maxpool(feat1)
        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'
        ), strict=False)
    return model

# -------------------- Residual U-Net --------------------
class ResUNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResUNetUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x1, x2):
        x2_up = self.up(x2)
        x = torch.cat([x1, x2_up], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResUNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResUNet, self).__init__()
        self.backbone = resnet50(pretrained=pretrained)
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        self.up4 = ResUNetUp(in_filters[3], out_filters[3])
        self.up3 = ResUNetUp(in_filters[2], out_filters[2])
        self.up2 = ResUNetUp(in_filters[1], out_filters[1])
        self.up1 = ResUNetUp(in_filters[0], out_filters[0])

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(x)
        x = self.up4(feat4, feat5)
        x = self.up3(feat3, x)
        x = self.up2(feat2, x)
        x = self.up1(feat1, x)
        return self.final(x)

if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    model = ResUNet(num_classes=2, pretrained=False)
    y = model(x)
    print(y.shape) 