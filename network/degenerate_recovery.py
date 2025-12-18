
import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DegenerateNet(nn.Module):
    def __init__(self):
        super(DegenerateNet, self).__init__()
        # 输入通道为3（三通道可见光）
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.decoder = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # 输出单通道伪红外
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out


class RecoveryNet(nn.Module):
    def __init__(self):
        super(RecoveryNet, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.decoder = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # 输出三通道可见光
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out