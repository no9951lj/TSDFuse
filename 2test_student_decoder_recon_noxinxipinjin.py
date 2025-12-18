import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from network.teacher_student_plus import StudentEncoder, SimpleDecoder


# ---------------- Dataset ---------------- #
class FusionTestDataset(Dataset):
    def __init__(self, vi_dir, ir_dir, transform=None):
        self.vi_paths = sorted([os.path.join(vi_dir, f) for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg'))])
        self.ir_paths = sorted([os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform

    def __len__(self):
        return min(len(self.vi_paths), len(self.ir_paths))

    def __getitem__(self, idx):
        vi = Image.open(self.vi_paths[idx]).convert('RGB')
        ir = Image.open(self.ir_paths[idx]).convert('RGB')
        name = os.path.basename(self.vi_paths[idx])
        if self.transform:
            vi = self.transform(vi)
            ir = self.transform(ir)
        return vi, ir, name


# ---------------- Save Image ---------------- #
def save_reconstructed_image(tensor, path):
    """保存重构图像，处理单通道灰度图"""
    array = tensor.squeeze().cpu().numpy()  # 移除批次和通道维度
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)  # 归一化到0-255
    Image.fromarray(array).save(path)


# ---------------- Inference ---------------- #
@torch.no_grad()
def run_reconstruction_test(student_ckpt, decoder_ckpt, vi_dir, ir_dir, output_dir):
    # 创建输出目录，区分不同特征组合的重构结果
    os.makedirs(output_dir, exist_ok=True)

    # 原始重构结果目录
    vi_recon_dir = os.path.join(output_dir, "vi_shared_vi_diff")
    ir_recon_dir = os.path.join(output_dir, "ir_shared_ir_diff")

    # 新增特征组合目录
    vi_shared_ir_diff_dir = os.path.join(output_dir, "vi_shared_ir_diff")
    ir_shared_vi_diff_dir = os.path.join(output_dir, "ir_shared_vi_diff")

    # 创建所有目录
    for dir_path in [vi_recon_dir, ir_recon_dir, vi_shared_ir_diff_dir, ir_shared_vi_diff_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 模型加载
    student = StudentEncoder().cuda()
    decoder = SimpleDecoder().cuda()

    # 加载权重
    student.load_state_dict(torch.load(student_ckpt))
    decoder.load_state_dict(torch.load(decoder_ckpt))

    student.eval()
    decoder.eval()

    # 数据预处理
    transform = transforms.ToTensor()
    dataset = FusionTestDataset(vi_dir, ir_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"🚀 Starting reconstruction on {len(dataset)} pairs...")

    for batch_idx, (vi, ir, name) in enumerate(loader):
        vi, ir = vi.cuda(), ir.cuda()
        base_name = name[0].split('.')[0]  # 获取文件名（不含扩展名）

        # 提取特征
        f_shared_vi, f_diff_vi = student(vi)
        f_shared_ir, f_diff_ir = student(ir)

        # ---------------- 原始重构方式 ---------------- #
        # 可见光图像重构 (vi_shared + vi_diff)
        vi_recon_input = torch.cat([f_shared_vi, f_diff_vi], dim=1)
        vi_recon = decoder(vi_recon_input)
        vi_recon = (vi_recon - torch.min(vi_recon)) / (torch.max(vi_recon) - torch.min(vi_recon))
        save_reconstructed_image(vi_recon, os.path.join(vi_recon_dir, f"{base_name}.png"))

        # 红外图像重构 (ir_shared + ir_diff)
        ir_recon_input = torch.cat([f_shared_ir, f_diff_ir], dim=1)
        ir_recon = decoder(ir_recon_input)
        ir_recon = (ir_recon - torch.min(ir_recon)) / (torch.max(ir_recon) - torch.min(ir_recon))
        save_reconstructed_image(ir_recon, os.path.join(ir_recon_dir, f"{base_name}.png"))

        # ---------------- 新增特征组合重构 ---------------- #
        # 可见光共享特征 + 红外差异特征
        vi_shared_ir_diff_input = torch.cat([f_shared_vi, f_diff_ir], dim=1)
        vi_shared_ir_diff_recon = decoder(vi_shared_ir_diff_input)
        vi_shared_ir_diff_recon = (vi_shared_ir_diff_recon - torch.min(vi_shared_ir_diff_recon)) / (
                    torch.max(vi_shared_ir_diff_recon) - torch.min(vi_shared_ir_diff_recon))
        save_reconstructed_image(vi_shared_ir_diff_recon, os.path.join(vi_shared_ir_diff_dir, f"{base_name}.png"))

        # 红外共享特征 + 可见光差异特征
        ir_shared_vi_diff_input = torch.cat([f_shared_ir, f_diff_vi], dim=1)
        ir_shared_vi_diff_recon = decoder(ir_shared_vi_diff_input)
        ir_shared_vi_diff_recon = (ir_shared_vi_diff_recon - torch.min(ir_shared_vi_diff_recon)) / (
                    torch.max(ir_shared_vi_diff_recon) - torch.min(ir_shared_vi_diff_recon))
        save_reconstructed_image(ir_shared_vi_diff_recon, os.path.join(ir_shared_vi_diff_dir, f"{base_name}.png"))

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            print(f"✅ Processed {batch_idx + 1}/{len(dataset)}: {base_name}")

    print(f"🎉 Reconstruction complete. Results saved to:")
    print(f"  - Visible reconstructed (vi_shared + vi_diff): {vi_recon_dir}")
    print(f"  - Infrared reconstructed (ir_shared + ir_diff): {ir_recon_dir}")
    print(f"  - Mixed reconstructed (vi_shared + ir_diff): {vi_shared_ir_diff_dir}")
    print(f"  - Mixed reconstructed (ir_shared + vi_diff): {ir_shared_vi_diff_dir}")


# ---------------- CLI入口 ---------------- #
if __name__ == "__main__":
    # 权重路径（请替换为实际训练好的权重）
    student_ckpt = "checkpoints_teacher_student_train1_plus_noxinxipinjin/best_student.pth"
    decoder_ckpt = "checkpoints_teacher_student_train1_plus_noxinxipinjin/best_decoder.pth"

    # 数据和输出路径
    vi_dir = "train_data/MSRS/test/vi"
    ir_dir = "train_data/MSRS/test/ir"
    output_dir = "results/reconstruction_noxinxipinjin_ceshi_test"  # 根目录，下分子目录保存四种重构结果

    run_reconstruction_test(student_ckpt, decoder_ckpt, vi_dir, ir_dir, output_dir)