import argparse
import torch.nn.functional as F
import cv2
import time
import numpy as np
from PIL import Image
import torch
import os
import torch.nn as nn
from torchvision.transforms import ToTensor
from utils_1.img_read_save import img_save

# -------------------- Network Definition -------------------- #

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
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Output: recovered visible
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out

# -------------------- Test Function -------------------- #

def test_recovery_net(model, image_path, output_dir=None, output_name=None):
    """
    使用恢复网络对单张图片进行测试
    model: 加载好的RecoveryNet模型
    image_path: 输入图片路径(红外图像)
    output_dir: 输出图片保存目录，若为None则返回numpy数组
    output_name: 输出图片文件名(不含扩展名)，若为None则使用输入文件名
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 加载输入图片
    image = Image.open(image_path).convert('L')  # 确保为灰度图
    
    # 预处理（转换为张量并添加batch维度）
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    # 推理
    with torch.no_grad():
        output_tensor = model(image_tensor)
    
    # 处理输出
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_np = np.squeeze((output_tensor * 255).cpu().detach().numpy())
    
    # 对RGB图像，调整维度顺序 [C,H,W] -> [H,W,C]
    if output_np.ndim == 3:
        output_np = np.transpose(output_np, (1, 2, 0))
    
    # 检查并替换 NaN 和 Inf 值
    output_np = np.nan_to_num(output_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 确保像素值在 0-255 范围内
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)
    
    # 保存结果或返回numpy数组
    if output_dir:
        # 确保保存目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用输入文件名作为默认输出名
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存图像
        try:
            img_save(output_np, output_name, output_dir)
            print(f"Test result saved to {os.path.join(output_dir, output_name)}.png")
            return None
        except Exception as e:
            print(f"Failed to save image: {e}")
            return output_np
    else:
        return output_np

def batch_test_recovery_net(model, input_dir, output_dir, prefix="recovered_"):
    """
    批量测试目录中的所有图片
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(image_files)} images to process...")
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        output_name = prefix + os.path.splitext(img_file)[0]
        
        try:
            test_recovery_net(model, img_path, output_dir, output_name)
            print(f"[{i+1}/{len(image_files)}] Processed: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Batch testing completed. {len(image_files)} images processed.")

# -------------------- Main Function -------------------- #

if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="Test Recovery Network")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_recover.pth", help="Path to the trained RecoveryNet model")
    parser.add_argument("--input_path", type=str, default="./degenerate_recovery_results/ir_wei", help="Path to input image or directory (infrared images)")
    parser.add_argument("--output_path", type=str, default="./degenerate_recovery_results/vi_recovered", help="Path to save output image or directory")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize image to (width height), e.g., --resize 512 512")
    parser.add_argument("--batch", action="store_true", default=True, help="Process all images in directory (if input_path is a directory)")
    args = parser.parse_args()
    
    # 设置数据转换
    transform = ToTensor()
    
    # 初始化模型
    recovery_net = RecoveryNet().cuda()
    
    # 加载模型权重
    try:
        recovery_net.load_state_dict(torch.load(args.model_path))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # 确保模型处于评估模式
    recovery_net.eval()
    
    # 处理输入路径
    if os.path.isfile(args.input_path):
        # 处理单张图片
        output_dir = args.output_path
        if output_dir is None:
            # 默认输出到同目录
            output_dir = os.path.dirname(args.input_path)
        
        # 执行测试
        start_time = time.time()
        test_recovery_net(
            model=recovery_net,
            image_path=args.input_path,
            output_dir=output_dir
        )
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
    
    elif os.path.isdir(args.input_path):
        # 批量处理目录中的所有图片
        output_dir = args.output_path
        if output_dir is None:
            # 默认输出到同级目录下的"recovered_results"文件夹
            output_dir = os.path.join(os.path.dirname(args.input_path), "recovered_results")
        
        # 执行批量测试
        start_time = time.time()
        batch_test_recovery_net(
            model=recovery_net,
            input_dir=args.input_path,
            output_dir=output_dir
        )
        end_time = time.time()
        print(f"Total batch processing time: {end_time - start_time:.4f} seconds")
    
    else:
        print("Error: input_path must be a valid image file or a directory")
        parser.print_help()
        exit(1)
    
    print("Testing completed successfully!")
