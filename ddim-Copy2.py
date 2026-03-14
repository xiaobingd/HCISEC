import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
from data_loader import DataLoader
from tools import intelligent_slice_and_align, normalize_mel_spectrogram, normalize_magnitude

# 引入优化后的模型
from model import BiometricUNet64
import torch
torch.cuda.empty_cache()
# ==========================================
#  1. SSIM 损失函数实现 (保持不变)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2.0
        img2 = (img2 + 1) / 2.0
        
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

# ==========================================
#  2. 全局配置与路径
# ==========================================
VIS_DIR = "./results_acc_bio/vis-ddiimp2ke" 
MODEL_DIR = "./results_acc_bio/models-ddimimp2ke"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
TIMESTEPS = 1000      # 训练时的总步数
BETA_START = 0.0001
BETA_END = 0.01
BATCH_SIZE = 8
EPOCHS = 1000
WARMUP_EPOCHS = 50 
LAMBDA_CYCLE = 0
CYCLE_CUTOFF = 500 

LAMBDA_SSIM = 0.4

# 🟢 DDIM 专用参数 (新增)
DDIM_TIMESTEPS = 50  # 采样步数 (加速：50步即可模拟500步的效果)
DDIM_ETA = 0.0       # eta=0 为纯 DDIM (确定性采样)，eta=1 为 DDPM

# ==========================================
#  3. Diffusion 数学工具 (含 DDIM 实现)
# ==========================================
# 定义 Beta Schedule
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 辅助函数：从列表中按 t 索引取值
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# 训练时用的前向加噪过程 (DDPM/DDIM 训练通用)
def forward_diffusion_sample(x_0, t, device):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# 用于 SSIM Loss 计算的 x0 预测函数
def predict_start_from_noise(x_t, t, noise):
    """从 x_t 和预测噪声反推 x_0"""
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return ((x_t - sqrt_one_minus_alphas_cumprod_t * noise) / (sqrt_alphas_cumprod_t + 1e-8))


def save_biometric_comparison(real_acc, fake_audio, rec_acc, real_audio, epoch, save_dir):
    def to_img(x): return (x[0, 0].cpu().numpy() + 1) / 2.0
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].imshow(to_img(real_acc), origin='lower', cmap='jet', aspect='auto',vmin=0.0, vmax=1.0)
    axs[0, 0].set_title('Real Acc (Source)')
    
    axs[0, 1].imshow(to_img(fake_audio), origin='lower', cmap='magma', aspect='auto',vmin=0.0, vmax=1.0)
    axs[0, 1].set_title('Generated Audio (DDIM)')
    
    axs[1, 0].imshow(to_img(rec_acc), origin='lower', cmap='jet', aspect='auto',vmin=0.0, vmax=1.0)
    axs[1, 0].set_title('Reconstructed Acc (DDIM)')
    
    axs[1, 1].imshow(to_img(real_audio), origin='lower', cmap='magma', aspect='auto',vmin=0.0, vmax=1.0)
    axs[1, 1].set_title('Real Audio (Target)')

    plt.suptitle(f"Biometric Check - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bio_check_epoch_{epoch}.png"))
    plt.close()
@torch.no_grad()
def stat(name, x):
    x_ = x.detach()
    print(name,
          "min", float(x_.min()),
          "max", float(x_.max()),
          "mean", float(x_.mean()),
          "nan", torch.isnan(x_).any().item(),
          "inf", torch.isinf(x_).any().item())
@torch.no_grad()
def ddim_sample_loop(model, condition, shape, ddim_steps=50, eta=0.0):
    """
    修复的 DDIM 采样 - 严格按照论文实现
    
    参考论文：Denoising Diffusion Implicit Models (ICLR 2021)
    核心公式：x_{t-1} = sqrt(α_{t-1}) * x0_pred + sqrt(1 - α_{t-1} - σ_t^2) * ε_pred + σ_t * z
    """
    device = next(model.parameters()).device
    b = shape[0]
    
    # 🔧 修复1：生成正确的时间步序列
    # 例如：TIMESTEPS=500, ddim_steps=50 -> [499, 490, 481, ..., 10, 0]
    # 注意：必须包含 0，否则最后一步无法正确处理
    step = TIMESTEPS // ddim_steps
    time_steps = list(range(TIMESTEPS - 1, -1, -step))
    if time_steps[-1] != 0:
        time_steps.append(0)  # 确保最后一步是 t=0
    
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.long)
    
    # 初始化为纯噪声
    img = torch.randn(shape, device=device)
    for i in range(len(time_steps) - 1):
        t = time_steps[i]
        t_next = time_steps[i + 1]

        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        eps_pred = model(img, t_batch, condition)

        alpha_t = alphas_cumprod[t]
        alpha_t_next = alphas_cumprod[t_next]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # 🔥 x0 prediction (只算一次)
        x0_pred = (img - sqrt_one_minus_alpha_t * eps_pred) / (sqrt_alpha_t + 1e-8)

        if t_next > 0:
            # DDIM update
            sigma_t = (
                eta
                * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t))
                * torch.sqrt(1 - alpha_t / alpha_t_next)
            )

            noise = torch.randn_like(img) if eta > 0 else 0.0

            img = (
                torch.sqrt(alpha_t_next) * x0_pred
                + torch.sqrt(1 - alpha_t_next - sigma_t**2) * eps_pred
                + sigma_t * noise
            )
        else:
            # 🔒 最后一步：只在这里 clamp
            img = x0_pred.clamp(-1, 1)
    return img
#     for i in tqdm(range(len(time_steps) - 1), desc='DDIM Sampling', leave=False):
#         t = time_steps[i]
#         t_next = time_steps[i + 1]
        
#         # 🔧 修复2：统一为 batch tensor
#         t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        
#         # 预测噪声
#         eps_pred = model(img, t_batch, condition)
#         eps_pred = torch.nan_to_num(eps_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)

        
#         # 🔧 修复3：获取 alpha 值（确保 shape 一致）
#         alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
#         alpha_t_next = alphas_cumprod[t_next].view(1, 1, 1, 1)
        
#         # 🔧 修复4：数值稳定的 x0 预测
#         sqrt_alpha_t = alpha_t.sqrt()
#         sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
        
#         alpha_bar_t = alpha_t
#         alpha_bar_prev = alpha_t_next

        
#         # x0 = (x_t - sqrt(1-α_t) * ε) / sqrt(α_t)
#         # x0_pred = (img - sqrt_one_minus_alpha_t * eps_pred) / (sqrt_alpha_t + 1e-8)
#         # x0_pred = x0_pred.clamp(-1.0, 1.0)
        
#         den = torch.clamp(sqrt_alpha_t, min=1e-4)
        
#         x0_pred = (img - sqrt_one_minus_alpha_t * eps_pred) / den
#         if t_next == 0:
#             img = x0_pred.clamp(-1, 1)

        
#         # 🔧 修复5：正确的 DDIM 公式
#         # σ_t = η * sqrt((1-α_{t-1})/(1-α_t)) * sqrt(1 - α_t/α_{t-1})
#         if eta > 0 and t_next > 0:
#             sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
#                           torch.sqrt(1 - alpha_bar_prev / alpha_bar_t)

#         else:
#             sigma_t = 0.0
        
#         # c1 = sqrt(α_{t-1})
#         c1 = alpha_t_next.sqrt()
        
#         # c2 = sqrt(1 - α_{t-1} - σ_t^2)
#         c2_squared = 1 - alpha_t_next - sigma_t ** 2
#         c2 = torch.sqrt(torch.clamp(c2_squared, min=0.0))  # 🔥 防止负数
        
#         # 更新
#         if t_next > 0:
#             noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
#             img = c1 * x0_pred + c2 * eps_pred + sigma_t * noise
#         else:
#             # 🔥 最后一步 (t=0)，直接输出 x0
#             img = x0_pred
    



# ==========================================
#  4. 主训练循环
# ==========================================
def train():
    model_a2u = BiometricUNet64(input_dim=1, condition_dim=1).to(device)
    model_u2a = BiometricUNet64(input_dim=1, condition_dim=1).to(device)

    opt_a2u = torch.optim.AdamW(model_a2u.parameters(), lr=1e-4)
    opt_u2a = torch.optim.AdamW(model_u2a.parameters(), lr=1e-4)

    loss_fn_mse = nn.MSELoss()
    loss_fn_ssim = SSIM(window_size=11).to(device)

    data_loader = DataLoader(
        # acc_path="./duibi/zrxqj_acc_sliced.npy",
        # audio_path="./duibi/zrxqj_audio_sliced.npy",
        acc_path="./duibi/ke_acc_c_sliced.npy",
        audio_path="./duibi/ke_audio_c_sliced.npy",
        test_size=0.2, normalize=False
    )
    
    train_size = len(data_loader.train_acc)
    num_batches = train_size // BATCH_SIZE 
    
    last_real_acc, last_real_audio = None, None
    best_val_loss = float('inf')
    avg_val_loss = float('inf')
    print(f"Start Training with SSIM & DDIM Sampler (Steps={DDIM_TIMESTEPS})...")

    for epoch in range(EPOCHS):
        model_a2u.train()
        model_u2a.train()
        
        pbar = tqdm(data_loader.get_batch_iter(BATCH_SIZE, dataset='train'), total=num_batches)
        total_loss = 0
        step_count = 0
        
        current_lambda_cycle = 0.0 if epoch < WARMUP_EPOCHS else LAMBDA_CYCLE
        
        for acc_batch_np, audio_batch_np in pbar:
            step_count += 1
            bs = acc_batch_np.shape[0]
            
            # --- 1. 数据预处理 ---
            acc_list, audio_list = [], []
            for i in range(bs):
                mel, mag = intelligent_slice_and_align(audio_batch_np[i], acc_batch_np[i])
                mel = normalize_mel_spectrogram(mel) * 2.0 - 1.0
                mag = normalize_magnitude(mag) * 2.0 - 1.0
                if torch.isnan(mel).any() or torch.isnan(mag).any(): continue
                audio_list.append(mel)
                acc_list.append(mag)
            
            if len(audio_list) == 0: continue
            
            real_audio = torch.stack(audio_list).to(device)
            real_acc = torch.stack(acc_list).to(device)
            
            if last_real_acc is None:
                last_real_acc = real_acc[0:1]
                last_real_audio = real_audio[0:1]
            
            current_bs = real_audio.shape[0]
            t = torch.randint(0, TIMESTEPS, (current_bs,), device=device).long()

            # --- 2. Phase 1: Diffusion Training (MSE + SSIM) ---
            # 注意：DDIM 的训练过程和 DDPM 是一模一样的，都是优化噪声预测
            
            # ==========================
            # 2.1 Acc -> Audio
            # ==========================
            x_t_audio, noise_audio = forward_diffusion_sample(real_audio, t, device)
            eps_pred_audio = model_a2u(x_t_audio, t, condition=real_acc)
            
            # MSE Loss
            loss_mse_a2u = loss_fn_mse(eps_pred_audio, noise_audio).mean()
            
            # SSIM Loss (利用预测的 x0)
            pred_audio_0 = predict_start_from_noise(x_t_audio, t, eps_pred_audio).clamp(-1.0, 1.0)
            loss_ssim_a2u = 1.0 - loss_fn_ssim(pred_audio_0, real_audio)
            
            loss_total_a2u = loss_mse_a2u + (LAMBDA_SSIM * loss_ssim_a2u)

            # ==========================
            # 2.2 Audio -> Acc
            # ==========================
            x_t_acc, noise_acc = forward_diffusion_sample(real_acc, t, device)
            eps_pred_acc = model_u2a(x_t_acc, t, condition=real_audio)
            
            loss_mse_u2a = loss_fn_mse(eps_pred_acc, noise_acc).mean()
            
            pred_acc_0 = predict_start_from_noise(x_t_acc, t, eps_pred_acc).clamp(-1.0, 1.0)
            loss_ssim_u2a = 1.0 - loss_fn_ssim(pred_acc_0, real_acc)
            
            loss_total_u2a = loss_mse_u2a + (LAMBDA_SSIM * loss_ssim_u2a)
            
            # --- 3. Optimization ---
            loss_final = loss_total_a2u + loss_total_u2a
            
            opt_a2u.zero_grad()
            opt_u2a.zero_grad()
            loss_final.backward()
            
            torch.nn.utils.clip_grad_norm_(model_a2u.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_u2a.parameters(), 1.0)
            
            opt_a2u.step()
            opt_u2a.step()
            
            total_loss += loss_final.item()
            pbar.set_description(f"Ep {epoch}| A2U: {loss_mse_a2u:.3f}+{loss_ssim_a2u:.3f} | Total: {loss_final.item():.3f}")
            
#             model_a2u.eval()
#             model_u2a.eval()
#             val_loss_total = 0.0
#             val_steps = 0
#             val_loss_a2u_total = 0.0
#             val_loss_u2a_total = 0.0

#             # 假设你的 data_loader.get_batch_iter 支持 dataset='test' 或 'val'
#             # 注意：验证时不使用梯度，不反向传播，为了速度只算 MSE
#         with torch.no_grad():
#             for acc_batch_np, audio_batch_np in data_loader.get_batch_iter(BATCH_SIZE, dataset='val'):
#                     # --- 数据预处理 (保持不变) ---
#                     acc_list, audio_list = [], []
#                     for i in range(len(acc_batch_np)):
#                         mel, mag = intelligent_slice_and_align(audio_batch_np[i], acc_batch_np[i])
#                         mel = normalize_mel_spectrogram(mel) * 2.0 - 1.0
#                         mag = normalize_magnitude(mag) * 2.0 - 1.0
#                         if torch.isnan(mel).any() or torch.isnan(mag).any(): continue
#                         audio_list.append(mel)
#                         acc_list.append(mag)

#                     if len(audio_list) == 0: continue

#                     real_audio = torch.stack(audio_list).to(device)
#                     real_acc = torch.stack(acc_list).to(device)
#                     bs = real_audio.shape[0]

#                     # 随机采样 t (验证集我们也看噪声预测的准确度)
#                     t = torch.randint(0, TIMESTEPS, (bs,), device=device).long()

#                     # ---------------------------
#                     # (A) 验证 A2U (Acc -> Audio)
#                     # ---------------------------
#                     x_t_audio, noise_audio = forward_diffusion_sample(real_audio, t, device)
#                     eps_pred_audio = model_a2u(x_t_audio, t, condition=real_acc)
#                     loss_a2u = loss_fn_mse(eps_pred_audio, noise_audio)
#                     val_loss_a2u_total += loss_a2u.item()

#                     # ---------------------------
#                     # (B) 验证 U2A (Audio -> Acc)
#                     # ---------------------------
#                     x_t_acc, noise_acc = forward_diffusion_sample(real_acc, t, device)
#                     eps_pred_acc = model_u2a(x_t_acc, t, condition=real_audio)
#                     loss_u2a = loss_fn_mse(eps_pred_acc, noise_acc)
#                     val_loss_u2a_total += loss_u2a.item()

#                     val_steps += 1

#             # 计算平均 Loss
#             avg_loss_a2u = val_loss_a2u_total / max(val_steps, 1)
#             avg_loss_u2a = val_loss_u2a_total / max(val_steps, 1)

#             # 💡 核心策略：监控两个 Loss 的总和
#             current_total_val_loss = avg_loss_a2u + avg_loss_u2a

#             print(f"Epoch {epoch} | Val A2U: {avg_loss_a2u:.4f} | Val U2A: {avg_loss_u2a:.4f} | Total: {current_total_val_loss:.4f}")

#             # ==========================
#             # 3. 保存最佳模型 (基于总 Loss)
#             # ==========================
#             if current_total_val_loss < best_val_loss:
#                 best_val_loss = current_total_val_loss
#                 # 同时保存两个模型，确保它们是“最佳搭档”
#                 torch.save(model_a2u.state_dict(), os.path.join(MODEL_DIR, "best_model_a2u.pth"))
#                 torch.save(model_u2a.state_dict(), os.path.join(MODEL_DIR, "best_model_u2a.pth"))
#                 print(f"New best models saved at epoch {epoch} (Total Val Loss: {best_val_loss:.4f})")


        # --- 4. Logging & Saving (使用 DDIM 采样) ---
        if (epoch + 1) % 10 == 0:
            print(f"Generating Visualization with DDIM ({DDIM_TIMESTEPS} steps)...")
            model_a2u.eval()
            model_u2a.eval()
            shape = (1, 1, 64, 64)
            if last_real_acc is not None:
                # 🟢 替换为 DDIM 采样函数
                fake_audio = ddim_sample_loop(model_a2u, condition=last_real_acc, shape=shape)
                rec_acc = ddim_sample_loop(model_u2a, condition=fake_audio, shape=shape)
                stat("fake_audio", fake_audio)
                stat("rec_acc", rec_acc)
                save_biometric_comparison(last_real_acc, fake_audio, rec_acc, last_real_audio, epoch, VIS_DIR)
        
        if (epoch + 1) % 20 == 0:
            torch.save(model_a2u.state_dict(), os.path.join(MODEL_DIR, f"model_a2u_ep{epoch}.pth"))
            torch.save(model_u2a.state_dict(), os.path.join(MODEL_DIR, f"model_u2a_ep{epoch}.pth"))

if __name__ == "__main__":
    train()
    
