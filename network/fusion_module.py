import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, radius=5, epsilon=0.01):
        """
        特征融合模块，用于深度学习模型中间层的特征融合

        参数:
            in_channels: 输入特征的通道数
            radius: 引导滤波半径
            epsilon: 引导滤波正则化参数
        """
        super(FeatureFusion, self).__init__()
        self.radius = radius
        self.epsilon = epsilon

        # 创建平均滤波器卷积核
        kernel_size = 2 * radius + 1
        avg_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.register_buffer('avg_kernel', avg_kernel)

        # 用于显著性计算的卷积层
        self.saliency_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def _guided_filter(self, I, p):
        """
        实现引导滤波算法

        参数:
            I: 引导特征 (BCHW格式)
            p: 待滤波特征 (BCHW格式)

        返回:
            q: 滤波后的特征
        """
        # 扩展卷积核以匹配通道数
        avg_kernel = self.avg_kernel.repeat(p.size(1), 1, 1, 1)

        # 计算均值
        mean_I = F.conv2d(I, avg_kernel, padding=self.radius, groups=I.size(1))
        mean_p = F.conv2d(p, avg_kernel, padding=self.radius, groups=p.size(1))
        mean_Ip = F.conv2d(I * p, avg_kernel, padding=self.radius, groups=p.size(1))

        # 计算协方差
        cov_Ip = mean_Ip - mean_I * mean_p

        # 计算方差
        mean_II = F.conv2d(I * I, avg_kernel, padding=self.radius, groups=I.size(1))
        var_I = mean_II - mean_I * mean_I

        # 计算线性系数a和b
        a = cov_Ip / (var_I + self.epsilon)
        b = mean_p - a * mean_I

        # 计算a和b的均值
        mean_a = F.conv2d(a, avg_kernel, padding=self.radius, groups=a.size(1))
        mean_b = F.conv2d(b, avg_kernel, padding=self.radius, groups=b.size(1))

        # 计算输出
        q = mean_a * I + mean_b
        return q

    def forward(self, feature1, feature2):
        """
        前向传播，融合两个特征图

        参数:
            feature1: 第一个特征图 (BCHW格式)
            feature2: 第二个特征图 (BCHW格式)

        返回:
            fused_feature: 融合后的特征图 (BCHW格式)
        """
        # 确保输入特征尺寸相同
        assert feature1.size() == feature2.size(), "两个特征图的尺寸必须相同"

        # 计算显著性图
        saliency1 = torch.sigmoid(self.saliency_conv(feature1))
        saliency2 = torch.sigmoid(self.saliency_conv(feature2))

        # 计算显著性差异
        diff1 = saliency1 - saliency2
        diff2 = saliency2 - saliency1

        # 使用步函数生成二值权重
        weight1 = torch.where(diff1 > 0, torch.ones_like(diff1), torch.zeros_like(diff1))
        weight2 = torch.where(diff2 > 0, torch.ones_like(diff2), torch.zeros_like(diff2))

        # 使用引导滤波优化权重
        guide1 = torch.mean(feature1, dim=1, keepdim=True)
        guide2 = torch.mean(feature2, dim=1, keepdim=True)

        refined_weight1 = self._guided_filter(guide1, weight1)
        refined_weight2 = self._guided_filter(guide2, weight2)

        # 归一化权重
        weight_sum = refined_weight1 + refined_weight2
        refined_weight1 = refined_weight1 / (weight_sum + 1e-8)
        refined_weight2 = refined_weight2 / (weight_sum + 1e-8)

        # 扩展权重维度以匹配特征图
        refined_weight1 = refined_weight1.repeat(1, feature1.size(1), 1, 1)
        refined_weight2 = refined_weight2.repeat(1, feature2.size(1), 1, 1)

        # 加权融合
        fused_feature = feature1 * refined_weight1 + feature2 * refined_weight2

        return fused_feature


def calculate_saliency(feature, method='norm', window_size=3, eps=1e-8):
    """修正多维度最大值计算，兼容旧版PyTorch"""
    if method == 'norm':
        # 使用L2范数作为显著性（对多通道特征更合理）
        saliency_map = torch.norm(feature, p=2, dim=1, keepdim=True)

    elif method == 'local_var':
        # 计算局部方差（方差越大表示该区域变化越剧烈，显著性越高）
        mean = F.avg_pool2d(feature, kernel_size=window_size, stride=1, padding=window_size // 2)
        squared_mean = F.avg_pool2d(feature.pow(2), kernel_size=window_size, stride=1, padding=window_size // 2)
        var = squared_mean - mean.pow(2)
        saliency_map = torch.mean(var, dim=1, keepdim=True)

    elif method == 'gradient':
        # 计算梯度幅值（边缘区域显著性高）
        # 定义Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                               device=feature.device).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                               device=feature.device).reshape(1, 1, 3, 3)

        # 扩展为多通道卷积核
        if feature.shape[1] > 1:
            sobel_x = sobel_x.expand(feature.shape[1], 1, 3, 3).contiguous()
            sobel_y = sobel_y.expand(feature.shape[1], 1, 3, 3).contiguous()

        # 计算梯度
        grad_x = F.conv2d(feature, sobel_x, padding=1, groups=feature.shape[1])
        grad_y = F.conv2d(feature, sobel_y, padding=1, groups=feature.shape[1])

        # 计算梯度幅值
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
        saliency_map = torch.mean(grad_mag, dim=1, keepdim=True)

    else:
        raise ValueError(f"Unsupported saliency method: {method}")

    # 关键修正：将 dim=[2,3] 拆分为分步计算（先按dim=3，再按dim=2）
    max_val = saliency_map
    # 先在宽度维度（dim=3）取最大值
    max_val = torch.max(max_val, dim=3, keepdim=True)[0]
    # 再在高度维度（dim=2）取最大值
    max_val = torch.max(max_val, dim=2, keepdim=True)[0]

    # 同理计算最小值
    min_val = saliency_map
    min_val = torch.min(min_val, dim=3, keepdim=True)[0]
    min_val = torch.min(min_val, dim=2, keepdim=True)[0]

    # 归一化到 [0, 1] 范围
    saliency_map = (saliency_map - min_val) / (max_val - min_val + eps)

    return saliency_map


def dynamic_weighted_fusion(f_diff_ir, f_diff_vi, saliency_method='norm', eps=1e-8):
    """基于显著性的动态加权融合（兼容旧版PyTorch）"""
    # 计算两种特征的显著性图
    saliency_ir = calculate_saliency(f_diff_ir, method=saliency_method)
    saliency_vi = calculate_saliency(f_diff_vi, method=saliency_method)

    # 计算动态权重（确保权重之和为1）
    weight_ir = saliency_ir / (saliency_ir + saliency_vi + eps)
    weight_vi = 1.0 - weight_ir

    # 加权融合
    fused_feature = weight_ir * f_diff_ir + weight_vi * f_diff_vi

    return fused_feature