import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision import models
import kornia  # 用于 SSIM 和 Sobel
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a smooth L1 variant)"""

    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.epsilon ** 2)
        return loss.mean()

def information_bottleneck_loss(mu, logvar, recon, target, beta=1e-3):
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return recon_loss + beta * kl_div


def cosine_difference_loss(vi_private, ir_private):
    """使用余弦距离最大化特征差异"""
    # 将特征展平为二维张量 (batch_size, feature_dim)
    vi_flat = vi_private.view(vi_private.size(0), -1)
    ir_flat = ir_private.view(ir_private.size(0), -1)

    # 计算余弦相似度（范围 [-1, 1]）
    cos_sim = F.cosine_similarity(vi_flat, ir_flat, dim=1)

    # 最大化差异 = 最小化相似度
    return torch.mean(cos_sim)

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        grad_x = F.conv2d(x, self.weight_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.weight_y.to(x.device), padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.sobel = Sobelxy()

    def forward(self, img1, img2):
        return F.l1_loss(self.sobel(img1), self.sobel(img2))


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)


# 假设你的perceptual_loss是基于VGG的，修改其forward方法
# 首先确保你已经定义了CharbonnierLoss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt(torch.pow(x - y, 2) + self.eps).mean()

# 修复PerceptualLoss类
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化VGG特征提取器（根据你的实际代码补充）
        self.vgg = self._init_vgg()  # 假设你有一个初始化VGG的方法
        # 定义charb属性（CharbonnierLoss实例）
        self.charb = CharbonnierLoss()  # 关键修复：添加这一行

    def _init_vgg(self):
        # 根据你的实际实现补充VGG特征提取器的初始化
        # 例如：
        vgg = models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg

    def forward(self, x, y):
        # 处理单通道转3通道（之前提到的修复）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        # 现在可以正常使用self.charb了
        return self.charb(self.vgg(x), self.vgg(y))