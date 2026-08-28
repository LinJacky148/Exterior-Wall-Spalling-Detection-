# 無CBAM
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --------------------- DenseNet Backbone ---------------------
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# --------------------- ASPP Module ---------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18), p_drop=0.0):
        super().__init__()
        d0, d1, d2, d3 = dilations

        # 1x1
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 atrous
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=d1, dilation=d1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=d2, dilation=d2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=d3, dilation=d3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        )

    def forward(self, x):
        size = x.shape[2:]

        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)

        xp = self.image_pool(x) # 1x1
        xp = F.interpolate(xp, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x0, x1, x2, x3, xp], dim=1)
        return self.project(x)
    
# ---------------- Full Model (fullres-only) ----------------
class DenseNet121_DeeplabV3Plus_Unet_4block1_1(nn.Module):
    """
    fullres-only：
      原圖 -> ASPP(256通道) -> 做3次下採樣(到 2h×2w) -> Bridge(MaxPool到 h×w)
      -> concat DenseBlock4 -> Decoder -> 上採樣回原尺寸
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):
        super().__init__()

        # ----- encoder stem -----
        self.encoder = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
        ]))

        num_features = num_init_features
        # DenseBlocks + Transitions
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.encoder.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.encoder.add_module(f'transition{i+1}', trans)
                num_features //= 2

        self.encoder.add_module('norm5', nn.BatchNorm2d(num_features))  # 最後通道預期為 1024

        #  新增Connection layer把原圖壓到8×8並升到256通道
        self.raw_stem = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=2, padding=1, bias=False),  # 256→128
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128→64
            nn.MaxPool2d(2),  # 64→32
            nn.MaxPool2d(2),  # 32→16
            nn.MaxPool2d(2),  # 16→8
        )
        #  在 8×8 上做 ASPP
        self.aspp_8 = ASPP(in_channels=256, out_channels=256, dilations=(1,6,12,18))

        # ----- decoder -----
        self.decode1_conv = nn.Sequential(
            nn.Conv2d(256 + 1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decode2_conv = nn.Sequential(
            nn.Conv2d(256 + 1024, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decode3_conv = nn.Sequential(
            nn.Conv2d(128 + 512, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decode4_conv = nn.Sequential(
            nn.Conv2d(64 + 256, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        img0 = x
        input_size = x.shape[2:]  # (H, W)

        # ----- Encoder & Skips -----
        skips = []
        for name, layer in self.encoder.named_children():
            x = layer(x)
            if name.startswith('denseblock'):
                skips.append(x)

        # ---- 原圖分支：直接在 8×8 做 ASPP ----
        x_8   = self.raw_stem(img0)  
        x_aspp = self.aspp_8(x_8)    
        x_aspp = F.interpolate(x_aspp, scale_factor=2, mode='bilinear', align_corners=False)  # → (B,256,16,16)
       
        # ----- Decoder -----
        x = torch.cat([x_aspp, skips[-1]], dim=1)  # (B,256+1024,h,w)
        x = self.decode1_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, skips[-2]], dim=1)
        x = self.decode2_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, skips[-3]], dim=1)
        x = self.decode3_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, skips[-4]], dim=1)
        x = self.decode4_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


# ==== quick test ====
if __name__ == '__main__':
    net = DenseNet121_DeeplabV3Plus_Unet_4block1_1(num_classes=2)
    x = torch.randn(4, 3, 256, 256)
    y = net(x)
    print("out:", y.shape)