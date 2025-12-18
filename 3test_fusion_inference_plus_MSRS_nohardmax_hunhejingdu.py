# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

# 你项目里的模块
from network.teacher_student_plus import StudentEncoder, SimpleDecoder
from utils_1.img_read_save import img_save
from network.fusion_module import FeatureFusion

# ==========================
# 日志：同时打印 + 写文件
# ==========================
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "fusion_test.txt")
_log_f = open(LOG_PATH, "a", encoding="utf-8")


def log(msg: str):
    print(msg)
    _log_f.write(msg + "\n")
    _log_f.flush()


# ------------------------------------------------------------
# 环境/性能设置（向下兼容）
# ------------------------------------------------------------
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


def _amp_dtype():
    if torch.cuda.is_available():
        # 能用 bf16 就用 bf16
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    return None


AMP_DTYPE = _amp_dtype()


def get_autocast_ctx(device, amp_dtype=None):
    # 旧接口
    try:
        import torch.cuda.amp as _amp
        if device.type == "cuda" and hasattr(_amp, "autocast"):
            return _amp.autocast(enabled=True)
    except Exception:
        pass

    # 新接口
    if hasattr(torch, "autocast"):
        kwargs = {"device_type": "cuda" if device.type == "cuda" else "cpu"}
        if amp_dtype is not None:
            try:
                return torch.autocast(dtype=amp_dtype, **kwargs)
            except TypeError:
                return torch.autocast(**kwargs)
        else:
            return torch.autocast(**kwargs)

    return nullcontext()


# ------------------------------------------------------------
# 数据集
# ------------------------------------------------------------
class FusionTestDataset(Dataset):
    def __init__(self, ir_dir, vi_dir, transform=None):
        self.ir_paths = sorted([
            os.path.join(ir_dir, f) for f in os.listdir(ir_dir)
            if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg', 'tif', 'tiff'))
        ])
        self.vi_paths = sorted([
            os.path.join(vi_dir, f) for f in os.listdir(vi_dir)
            if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg', 'tif', 'tiff'))
        ])
        self.transform = transform

    def __len__(self):
        return min(len(self.ir_paths), len(self.vi_paths))

    def __getitem__(self, idx):
        ir = Image.open(self.ir_paths[idx]).convert('RGB')
        vi = Image.open(self.vi_paths[idx]).convert('RGB')
        name = os.path.basename(self.ir_paths[idx])
        if self.transform:
            ir = self.transform(ir)
            vi = self.transform(vi)
        return ir, vi, name


# ------------------------------------------------------------
# 私有特征软融合（AMP 兼容）
# ------------------------------------------------------------
def softmax_fuse_private(f_ir, f_vi, tau=0.5, eps=1e-6):
    # f_*: [B, C, H, W]
    s_ir = torch.sqrt((f_ir ** 2).mean(dim=1, keepdim=True) + eps)
    s_vi = torch.sqrt((f_vi ** 2).mean(dim=1, keepdim=True) + eps)

    w_ir = torch.exp(s_ir / tau)
    w_vi = torch.exp(s_vi / tau)
    w_sum = w_ir + w_vi + eps
    w_ir, w_vi = w_ir / w_sum, w_vi / w_sum

    # 做个轻量平滑
    kernel = torch.tensor(
        [[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]],
        device=f_ir.device,
        dtype=f_ir.dtype,
    )
    kernel = kernel / kernel.sum()
    k = kernel.view(1, 1, 3, 3)
    w_ir = F.conv2d(w_ir, k, padding=1)
    w_vi = F.conv2d(w_vi, k, padding=1)

    return w_ir * f_ir + w_vi * f_vi


# ------------------------------------------------------------
# 统计参数量
# ------------------------------------------------------------
def count_params_in_M(*models):
    total = 0
    for m in models:
        total += sum(p.numel() for p in m.parameters())
    return total / 1e6


# ------------------------------------------------------------
# 用于 FLOPs 的整体包装
# ------------------------------------------------------------
class WholeFusion(torch.nn.Module):
    def __init__(self, student, decoder, use_soft=True):
        super().__init__()
        self.student = student
        self.decoder = decoder
        self.use_soft = use_soft

    def forward(self, ir_rgb, vi_rgb):
        # 和下面 test_fusion 里的逻辑保持一致
        f_shared_ir, f_diff_ir = self.student(ir_rgb)
        f_shared_vi, f_diff_vi = self.student(vi_rgb)

        f_shared_fused = (f_shared_ir + f_shared_vi) / 2
        if self.use_soft:
            f_diff_fused = softmax_fuse_private(f_diff_ir, f_diff_vi, tau=0.6)
        else:
            f_diff_fused = (f_diff_ir + f_diff_vi) / 2

        fused_input = torch.cat([f_shared_fused, f_diff_fused], dim=1)
        recon = self.decoder(fused_input)
        return recon


# ------------------------------------------------------------
# 主推理逻辑（混合精度 + 性能统计 + 日志）
# ------------------------------------------------------------
def test_fusion(student_path, decoder_path,
                ir_dir, vi_dir, output_dir,
                batch_size=1, num_workers=4):

    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FusionTestDataset(ir_dir, vi_dir, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_mem = (device.type == 'cuda')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # 1) 加载模型
    student = StudentEncoder().to(device)
    decoder = SimpleDecoder().to(device)
    fusion_module = FeatureFusion(in_channels=256).to(device)  # 你原来就建了，也算进参数

    # channels-last
    try:
        student = student.to(memory_format=torch.channels_last)
        decoder = decoder.to(memory_format=torch.channels_last)
        fusion_module = fusion_module.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # 2) 读取权重
    student_state = torch.load(student_path, map_location=device)
    decoder_state = torch.load(decoder_path, map_location=device)
    student.load_state_dict(student_state)
    decoder.load_state_dict(decoder_state)

    student.eval()
    decoder.eval()
    fusion_module.eval()  # 虽然没用上，但先设 eval

    # 3) 参数量
    total_params_M = count_params_in_M(student, decoder, fusion_module)
    log(f"[MODEL] Total params (student+decoder+fusion): {total_params_M:.3f} M")

    # 4) 尝试算 FLOPs（整条前向）
    try:
        from thop import profile, clever_format
        whole = WholeFusion(student, decoder).to(device)
        dummy_ir = torch.randn(1, 3, 256, 256).to(device)
        dummy_vi = torch.randn(1, 3, 256, 256).to(device)
        flops, params = profile(whole, inputs=(dummy_ir, dummy_vi), verbose=False)
        flops_g = flops / 1e9
        log(f"[MODEL] Whole FLOPs (256x256): {flops_g:.3f} G")
    except Exception as e:
        log(f"[WARN] FLOPs not computed: {e}")

    # 5) 推理 + 计时
    total_time = 0.0
    total_imgs = 0

    with torch.no_grad():
        for ir, vi, names in loader:
            ir = ir.to(device, non_blocking=pin_mem)
            vi = vi.to(device, non_blocking=pin_mem)
            try:
                ir = ir.to(memory_format=torch.channels_last)
                vi = vi.to(memory_format=torch.channels_last)
            except Exception:
                pass

            start_t = time.time()
            # AMP
            with get_autocast_ctx(device, AMP_DTYPE):
                f_shared_ir, f_diff_ir = student(ir)
                f_shared_vi, f_diff_vi = student(vi)

                f_shared_fused = (f_shared_ir + f_shared_vi) / 2
                f_diff_fused = softmax_fuse_private(f_diff_ir, f_diff_vi, tau=0.6)

                fused_input = torch.cat([f_shared_fused, f_diff_fused], dim=1)
                recon = decoder(fused_input)
                output_tensor = torch.clamp(recon, 0, 1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_t = time.time()

            elapsed = end_t - start_t
            total_time += elapsed
            bsz = ir.shape[0]
            total_imgs += bsz

            # 保存
            out_cpu = output_tensor.detach().float().cpu()
            for i in range(bsz):
                name = names[i]
                x = out_cpu[i]
                if x.dim() == 3 and x.shape[0] == 1:
                    x = x.squeeze(0)
                arr = x.numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
                arr = (arr * 255).astype(np.uint8)
                if arr.ndim == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                img_save(arr, os.path.splitext(name)[0], output_dir)

            log(f"[SAVE] batch with {bsz} image(s)  time={elapsed:.4f}s")

    # 6) 统计
    if total_imgs > 0:
        avg_time = total_time / total_imgs
        fps = 1.0 / avg_time
    else:
        avg_time = 0.0
        fps = 0.0

    log(f"[RESULT] total_imgs={total_imgs}, avg_time={avg_time:.6f}s, FPS={fps:.2f}")
    log(f"[INFO] results saved to {output_dir}")

    _log_f.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Fusion Test (with params/FLOPs/log/AMP)"
    )
    parser.add_argument('--student_path', type=str,
                        default="./2checkpoints_teacher_student_train1_plus_noxinxipinjin/best_student.pth",
                        help='Path to trained student model')
    parser.add_argument('--decoder_path', type=str,
                        default="./3checkpoints_decoder3_only_aver/decoder_epoch20.pth",
                        help='Path to trained decoder model')
    parser.add_argument('--ir_dir', type=str, default="./test_data/MSRS/ir",
                        help='Directory of infrared test images')
    parser.add_argument('--vi_dir', type=str, default="./test_data/MSRS/vi",
                        help='Directory of visible test images')
    parser.add_argument('--output_dir', type=str,
                        default="./results/MSRS_0.6_ampcompat",
                        help='Directory to save fusion results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    args = parser.parse_args()

    test_fusion(
        student_path=args.student_path,
        decoder_path=args.decoder_path,
        ir_dir=args.ir_dir,
        vi_dir=args.vi_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
