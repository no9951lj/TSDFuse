import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from einops import rearrange
from network.loss import CharbonnierLoss
import numbers

# ------------------- 教师编码器（不下采样） ------------------- #
class TeacherEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet34(pretrained=True)

        # 魔改：去下采样
        base.conv1.stride = (1, 1)
        base.maxpool = nn.Identity()
        base.layer1[0].conv1.stride = (1, 1)
        base.layer2[0].conv1.stride = (1, 1)
        base.layer3[0].conv1.stride = (1, 1)

        # 修复残差连接的 downsample 分支 stride
        for layer in [base.layer1, base.layer2, base.layer3]:
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (1, 1)

        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.layer1, base.layer2, base.layer3
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)  # shape: [B, 256, H, W]


# ------------------- 学生编码器（不下采样） ------------------- #
class StudentEncoder(nn.Module):
    def __init__(self, shared_channels=256, diff_channels=256):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.shared_proj = nn.Sequential(
            nn.Conv2d(256, shared_channels, 1),
            nn.BatchNorm2d(shared_channels), nn.ReLU()
        )

        self.diff_proj = nn.Sequential(
            nn.Conv2d(256, diff_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(diff_channels, diff_channels, 3, padding=1),
            nn.ReLU()
        )

        self.mu_head = nn.Conv2d(diff_channels, diff_channels, 3, padding=1)
        self.logvar_head = nn.Conv2d(diff_channels, diff_channels, 3, padding=1)

    def forward(self, x):
        feat = self.base(x)
        F_shared = self.shared_proj(feat)

        F_diff = self.diff_proj(feat)
        # mu = self.mu_head(diff_feat)
        # logvar = self.logvar_head(diff_feat)
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # F_diff = mu + eps * std
        # return F_shared, F_diff, mu, logvar
        return F_shared, F_diff

# ------------------- 解码器 ------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- SE 注意力模块 ------------------- #
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x)
        return x * weight

# ------------------- ResBlock ------------------- #
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x)

# ------------------- Hybrid Decoder ------------------- #
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super(SimpleDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(256),
            SEBlock(256),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(128),
            SEBlock(128),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)

