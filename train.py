import torch
import os
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from data_loader import DataLoader
from model_drop import (GeneratorAudioToAcc, GeneratorAccToAudio,
                        DiscriminatorAudio, DiscriminatorAcc)
from loss_new import WGANLoss, generator_loss
from tools import (normalize_mel_spectrogram, compute_gradient_penalty,
                   normalize_magnitude, intelligent_slice_and_align)
from torch.optim import lr_scheduler

# =====================
# 1. 基础配置
# =====================
model_dir = "./models_0302ke11"
log_dir   = "./logs_0302ke11"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir,   exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# 2. 数据加载
# =====================
data_loader = DataLoader(
    acc_path   = "./ke_acc_c_sliced.npy",
    audio_path = "./ke_audio_c_sliced.npy",
    test_size    = 0.2,
    random_state = 42,
    normalize    = True,
)

# =====================
# 3. 模型初始化
# =====================
G_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1).to(device)
G_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1).to(device)
D_audio     = DiscriminatorAudio(input_dim=1, base=64, n_scales=1,
                                  use_spectral_norm=True).to(device)
D_acc       = DiscriminatorAcc(input_dim=1,   base=64, n_scales=1,
                                use_spectral_norm=True).to(device)

try:
    summary(G_audio2acc, input_size=(1, 64, 64),
            device=str(device).split(':')[0])
except Exception as e:
    print(f"Summary error: {e}")

# =====================
# 4. 辅助损失函数
# =====================
class FrequencyAwareLoss(nn.Module):
    """低频加权MSE，自动适配输入频率维度"""
    def __init__(self, low_freq_ratio=0.2):
        super().__init__()
        self.low_freq_ratio   = low_freq_ratio
        self._cached_weight   = None
        self._cached_freq_bins = None

    def _get_weights(self, freq_bins, device):
        if self._cached_freq_bins != freq_bins:
            w = torch.ones(freq_bins)
            low_end = int(freq_bins * self.low_freq_ratio)
            w[:low_end] = 3.0
            w[low_end:] = 0.5
            self._cached_weight    = w.view(1, 1, -1, 1).to(device)
            self._cached_freq_bins = freq_bins
        return self._cached_weight

    def forward(self, pred, target):
        freq_bins = pred.shape[2]
        weights   = self._get_weights(freq_bins, pred.device)
        diff      = (pred - target) ** 2
        return (diff * weights).mean()


class SpectralSmoothLoss(nn.Module):
    """惩罚频率方向高频振荡，只用于 acc 域"""
    def forward(self, x):
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]   # 频率轴
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]   # 时间轴
        return diff_h.abs().mean() * 2.0 + diff_w.abs().mean()


class FeatureMatchingLoss(nn.Module):
    """
    从判别器中间层提取特征计算L1距离。
    比像素级L2更能保留谐波等高频结构细节。
    """
    def forward(self, real_feats, fake_feats):
        loss = 0.0
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
        return loss / max(len(real_feats), 1)

# train.py 中添加转换函数
def to_neg1_pos1(tensor):
    """将 [0,1] 范围的 tensor 转换到 [-1,1]"""
    return tensor * 2.0 - 1.0

def from_neg1_pos1(tensor):
    """将 [-1,1] 范围的 tensor 转换回 [0,1]（可视化用）"""
    return (tensor + 1.0) * 0.5

freq_loss_fn   = FrequencyAwareLoss(low_freq_ratio=0.2).to(device)
spectral_smooth = SpectralSmoothLoss().to(device)
feat_match_loss = FeatureMatchingLoss().to(device)
# =====================
# 11. 可视化辅助函数
# =====================
def _plot_domain(orig, pred, cycl, title, epoch, log_dir):
    """
    orig / pred / cycl : numpy [B, F, T]
    绘制前 min(B,2) 个样本的 原始/预测/循环恢复 对比图
    """
    n = min(orig.shape[0], 2)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = [f'Original {title}', f'Predicted {title}', f'Cycled {title}']
    data_rows  = [orig, pred, cycl]

    for row, (data, lbl) in enumerate(zip(data_rows, row_labels)):
        for col in range(n):
            im = axes[row, col].imshow(
                data[col], aspect='auto', origin='lower',
                cmap='viridis',
                vmin=data[col].min(), vmax=data[col].max()
            )
            axes[row, col].set_title(f'{lbl} (Sample {col})', fontsize=10)
            axes[row, col].set_xlabel('Time Frame')
            axes[row, col].set_ylabel('Freq Bin')
            plt.colorbar(im, ax=axes[row, col])

    stage = '1' if epoch <= 30 else ('2' if epoch <= 100 else '3')
    plt.suptitle(
        f'{title} Domain @ Epoch {epoch} | Stage {stage}',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(log_dir, f'vis_{title.lower()}_ep{epoch}.png'),
        dpi=150
    )
    plt.close()
# =====================
# 5. 动态权重调度
# =====================
# def get_dynamic_lambda(epoch):
#     """
#     Stage1 (0-29)  : 重构优先，网络先学会基本映射
#     Stage2 (30-99) : 线性过渡，逐步引入 cycle / feat_match
#     Stage3 (100+)  : 正常 CycleGAN，feat_match 主导清晰度
#     """
#     if epoch < 30:
#         return dict(adv=0.5, rec=10.0, cycle=1.0, identity=1.0,
#                     freq=3.0, smooth=0.5, feat_match=2.0)
#     elif epoch < 100:
#         p = (epoch - 30) / 70.0          # 0→1
#         return dict(
#             adv        = 0.5  + 0.5  * p,   # 0.1  → 1.0
#             rec        = 10.0 - 5.0  * p,   # 10.0 → 5.0
#             cycle      = 1.0  + 7.0  * p,   # 1.0  → 8.0
#             identity   = 1.0  - 0.9  * p,   # 2.0  → 1.0
#             freq       = 3.0  - 1.0  * p,   # 3.0  → 2.0
#             smooth     = 0.5  - 0.2  * p,   # 0.5  → 0.1
#             feat_match = 2.0  + 2.0  * p,   # 2.0  → 4.0
#         )
#     else:
#         return dict(adv=1.0, rec=5.0, cycle=8.0, identity=0.1,
#                     freq=2.0, smooth=0.3, feat_match=4.0)
def get_dynamic_lambda(epoch):
    """
    Stage1 (0-29)  : 重构优先，网络先学会基本映射
    Stage2 (30-99) : 线性过渡，逐步引入 cycle / feat_match
    Stage3 (100+)  : 正常 CycleGAN，feat_match 主导清晰度
    """
    if epoch < 50:
        return dict(adv=0.5, rec=10.0, cycle=1.0, identity=1.0,
                    freq=3.0, smooth=0.5, feat_match=2.0)
    elif epoch < 150:
        p = (epoch - 30) / 70.0          # 0→1
        return dict(
            adv        = 0.5  + 0.5  * p,   # 0.1  → 1.0
            rec        = 10.0 - 5.0  * p,   # 10.0 → 5.0
            cycle      = 1.0  + 7.0  * p,   # 1.0  → 8.0
            identity   = 1.0  - 0.9  * p,   # 2.0  → 1.0
            freq       = 3.0  - 1.0  * p,   # 3.0  → 2.0
            smooth     = 0.5  - 0.3  * p,   # 0.5  → 0.1
            feat_match = 2.0  + 2.0  * p,   # 2.0  → 4.0
        )
    else:
        return dict(adv=1.0, rec=5.0, cycle=8.0, identity=0.1,
                    freq=2.0, smooth=0.2, feat_match=4.0)
# =====================
# 6. 优化器与调度器
# =====================
lr_G  = 1e-4
lr_D  = 2e-4
beta1, beta2 = 0.5, 0.999

optimizer_G = torch.optim.Adam(
    list(G_audio2acc.parameters()) + list(G_acc2audio.parameters()),
    lr=lr_G, betas=(beta1, beta2)
)
optimizer_D_audio = torch.optim.Adam(D_audio.parameters(), lr=lr_D, betas=(beta1, beta2))
optimizer_D_acc   = torch.optim.Adam(D_acc.parameters(),   lr=lr_D, betas=(beta1, beta2))

lambda_gp              = 10.0
num_epochs             = 500
batch_size             = 16
warmup, N0, N1         = 50, 250, 50
total_epochs           = warmup + N0 + N1

def lr_lambda(epoch):
    if epoch < warmup:
        return (epoch + 1) / warmup
    e = epoch - warmup
    if e < N0:
        return 1.0
    return max(0.0, 1.0 - (e - N0) / float(N1))

scheduler_G       = lr_scheduler.LambdaLR(optimizer_G,       lr_lambda=lr_lambda)
scheduler_D_audio = lr_scheduler.LambdaLR(optimizer_D_audio, lr_lambda=lr_lambda)
scheduler_D_acc   = lr_scheduler.LambdaLR(optimizer_D_acc,   lr_lambda=lr_lambda)

wgan_loss = WGANLoss()

# =====================
# 7. 损失记录容器
# =====================
history = dict(
    d_audio=[], d_acc=[],
    g_a2acc=[], g_acc2a=[],
    cycle=[], freq_cycle=[],
    smooth=[], feat_match=[]
)
current_lambdas = {}

# =====================
# 8. 训练循环
# =====================
print("Starting Training Loop...")

for epoch in range(total_epochs):

    # ---- 获取本 epoch 动态权重 ----
    current_lambdas  = get_dynamic_lambda(epoch)
    lambda_adv       = current_lambdas['adv']
    lambda_rec       = current_lambdas['rec']
    lambda_cycle     = current_lambdas['cycle']
    lambda_identity  = current_lambdas['identity']
    lambda_freq      = current_lambdas['freq']
    lambda_smooth    = current_lambdas['smooth']
    lambda_feat      = current_lambdas['feat_match']

    if epoch in (0, 30, 100):
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Lambda weights →  {current_lambdas}")
        print(f"{'='*60}\n")

    # ---- Epoch 累计量 ----
    ep = {k: 0.0 for k in
          ['d_audio','d_acc','g_a2acc','g_acc2a',
           'cycle','freq_cycle','smooth','feat_match']}
    num_batches       = 0
    critic_iterations = 5 if epoch < 50 else 2

    for i, (acc_np, audio_np) in enumerate(
            data_loader.get_batch_iter(batch_size, dataset='train')):

        num_batches  += 1
        current_bs    = acc_np.shape[0]

        # -------------------------------------------------
        # A. 数据预处理
        # -------------------------------------------------
        audio_list, acc_list = [], []
        for b in range(current_bs):
            mel, mag = intelligent_slice_and_align(audio_np[b], acc_np[b])
            mel = normalize_mel_spectrogram(mel)
            mag = normalize_magnitude(mag)
            mel = to_neg1_pos1(mel)
            mag = to_neg1_pos1(mag)
            audio_list.append(mel)
            acc_list.append(mag)

            
        audio_mel = torch.stack(audio_list).to(device)   # [B,1,F,T]
        magnitude = torch.stack(acc_list).to(device)

        if i == 0 and epoch == 0:
            print(f"[Shape] audio_mel={audio_mel.shape}  magnitude={magnitude.shape}")

        # -------------------------------------------------
        # B. 判别器训练（WGAN-GP）
        # -------------------------------------------------
        for _ in range(critic_iterations):

            # --- D_audio ---
            optimizer_D_audio.zero_grad()
            with torch.no_grad():
                fake_audio_tmp = G_acc2audio(magnitude)
            real_v = D_audio(audio_mel)
            fake_v = D_audio(fake_audio_tmp.detach())
            d_audio_loss = (wgan_loss.discriminator_loss(real_v, fake_v)
                            + lambda_gp * compute_gradient_penalty(
                                D_audio, audio_mel, fake_audio_tmp, device))
            d_audio_loss.backward()
            optimizer_D_audio.step()

            # --- D_acc ---
            optimizer_D_acc.zero_grad()
            with torch.no_grad():
                fake_acc_tmp = G_audio2acc(audio_mel)
            real_v = D_acc(magnitude)
            fake_v = D_acc(fake_acc_tmp.detach())
            d_acc_loss = (wgan_loss.discriminator_loss(real_v, fake_v)
                          + lambda_gp * compute_gradient_penalty(
                              D_acc, magnitude, fake_acc_tmp, device))
            d_acc_loss.backward()
            optimizer_D_acc.step()

        ep['d_audio'] += d_audio_loss.item()
        ep['d_acc']   += d_acc_loss.item()

        # -------------------------------------------------
        # C. 生成器训练
        # -------------------------------------------------
        optimizer_G.zero_grad()

        # C-1. 前向
        fake_audio_mel = G_acc2audio(magnitude)   # Acc  → Fake Audio
        fake_acc_mag   = G_audio2acc(audio_mel)   # Audio → Fake Acc

        # C-2. 判别器打分（带中间特征）
        fake_audio_v, fake_audio_feats = D_audio(fake_audio_mel, return_feats=True)
        real_audio_v, real_audio_feats = D_audio(audio_mel,      return_feats=True)

        fake_acc_v,   fake_acc_feats   = D_acc(fake_acc_mag, return_feats=True)
        real_acc_v,   real_acc_feats   = D_acc(magnitude,    return_feats=True)

        # C-3. 循环一致性
        recovered_acc_mag   = G_audio2acc(fake_audio_mel)  # Acc→Audio→Acc
        recovered_audio_mel = G_acc2audio(fake_acc_mag)    # Audio→Acc→Audio

        # C-4. 身份映射
        identity_audio = G_acc2audio(audio_mel) if lambda_identity > 0 else None
        identity_acc   = G_audio2acc(magnitude)  if lambda_identity > 0 else None

        # C-5. 基础 CycleGAN 损失
        g_a2acc_total, _, _, g_a2acc_cyc, _ = generator_loss(
            fake_acc_v, magnitude, fake_acc_mag, recovered_acc_mag,
            identity_acc, lambda_adv, lambda_rec, lambda_cycle, lambda_identity
        )
        g_acc2a_total, _, _, g_acc2a_cyc, _ = generator_loss(
            fake_audio_v, audio_mel, fake_audio_mel, recovered_audio_mel,
            identity_audio, lambda_adv, lambda_rec, lambda_cycle, lambda_identity
        )

        # C-6. 特征匹配损失
        # audio 域权重 ×2
        # C-6. 特征匹配损失
        # audio 域权重 ×2（修复 Pred Mel 模糊的关键）
        feat_audio = feat_match_loss(real_audio_feats, fake_audio_feats) * 2.0
        feat_acc   = feat_match_loss(real_acc_feats,   fake_acc_feats)
        feat_total = feat_audio + feat_acc

        # C-7. 频率感知循环损失（只约束 acc 域）
        freq_cycle_loss = freq_loss_fn(recovered_acc_mag, magnitude)

        # C-8. 谱平滑损失（只约束 acc 域，不碰 audio 域）
        smooth_loss = (spectral_smooth(fake_acc_mag)
                     + spectral_smooth(recovered_acc_mag))

        # C-9. 汇总
        g_total    = (g_a2acc_total
                    + g_acc2a_total
                    + lambda_feat   * feat_total
                    + lambda_freq   * freq_cycle_loss
                    + lambda_smooth * smooth_loss)
        cycle_loss = g_a2acc_cyc + g_acc2a_cyc

        g_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(G_audio2acc.parameters()) + list(G_acc2audio.parameters()),
            max_norm=1.0
        )
        optimizer_G.step()

        # C-10. 累计
        ep['g_a2acc']      += g_a2acc_total.item()
        ep['g_acc2a']      += g_acc2a_total.item()
        ep['cycle']        += cycle_loss.item()
        ep['freq_cycle']   += freq_cycle_loss.item()
        ep['smooth']       += smooth_loss.item()
        ep['feat_match']   += feat_total.item()

        # C-11. Batch 日志
        if i % 50 == 0:
            stage = ('1-Rec' if epoch < 30
                     else ('2-Trans' if epoch < 100 else '3-Cycle'))
            print(
                f"[Ep {epoch:>3d}/{total_epochs} | Bt {i:>4d} | {stage}] "
                f"D: {d_audio_loss.item():.3f}/{d_acc_loss.item():.3f} | "
                f"G: {g_total.item():.3f} "
                f"Cyc: {cycle_loss.item():.3f} | "
                f"Feat: {feat_total.item():.3f} "
                f"(aud={feat_audio.item():.3f} acc={feat_acc.item():.3f}) | "
                f"Freq: {freq_cycle_loss.item():.3f} "
                f"Smo: {smooth_loss.item():.3f}"
            )

    # =====================
    # 9. Epoch 结束
    # =====================
    scheduler_G.step()
    scheduler_D_audio.step()
    scheduler_D_acc.step()

    # 计算平均并记录
    for k in ep:
        history[k].append(ep[k] / num_batches)

    lr_now = optimizer_G.param_groups[0]['lr']
    print(
        f"\n>>> Ep {epoch+1:>3d}/{total_epochs} | LR={lr_now:.2e} | "
        f"λ(adv={lambda_adv:.1f} rec={lambda_rec:.1f} "
        f"cyc={lambda_cycle:.1f} id={lambda_identity:.1f} "
        f"freq={lambda_freq:.1f} smo={lambda_smooth:.2f} "
        f"feat={lambda_feat:.1f})\n"
        f"    D_audio={history['d_audio'][-1]:.4f}  "
        f"D_acc={history['d_acc'][-1]:.4f}\n"
        f"    G_a2acc={history['g_a2acc'][-1]:.4f}  "
        f"G_acc2a={history['g_acc2a'][-1]:.4f}\n"
        f"    Cycle={history['cycle'][-1]:.4f}  "
        f"FreqCyc={history['freq_cycle'][-1]:.4f}  "
        f"Smooth={history['smooth'][-1]:.4f}  "
        f"FeatMatch={history['feat_match'][-1]:.4f}\n"
    )

    # =====================
    # 10. 定期保存
    # =====================
    if (epoch + 1) % 50 == 0:

        # ---------- 保存权重 ----------
        save_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch'                       : epoch + 1,
            'G_audio2acc_state_dict'      : G_audio2acc.state_dict(),
            'G_acc2audio_state_dict'      : G_acc2audio.state_dict(),
            'D_audio_state_dict'          : D_audio.state_dict(),
            'D_acc_state_dict'            : D_acc.state_dict(),
            'optimizer_G_state_dict'      : optimizer_G.state_dict(),
            'optimizer_D_audio_state_dict': optimizer_D_audio.state_dict(),
            'optimizer_D_acc_state_dict'  : optimizer_D_acc.state_dict(),
            'scheduler_G_state_dict'      : scheduler_G.state_dict(),
            'history'                     : history,
            'current_lambdas'             : current_lambdas,
        }, save_path)
        print(f"[Save] Checkpoint → {save_path}")

        # ---------- Loss 曲线 ----------
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))

        axes[0].set_title('Discriminator Loss')
        axes[0].plot(history['d_audio'], label='D_Audio', color='steelblue')
        axes[0].plot(history['d_acc'],   label='D_Acc',   color='darkorange')

        axes[1].set_title('Generator Loss')
        axes[1].plot(history['g_a2acc'], label='G_audio2acc', color='green')
        axes[1].plot(history['g_acc2a'], label='G_acc2audio', color='purple')

        axes[2].set_title('Cycle Loss')
        axes[2].plot(history['cycle'],      label='Cycle',    color='red',    lw=2)
        axes[2].plot(history['freq_cycle'], label='FreqCycle',color='tomato', lw=1.5)
        axes[2].plot(history['smooth'],     label='Smooth',   color='peru',   lw=1.5)

        axes[3].set_title('Feature Matching Loss')
        axes[3].plot(history['feat_match'], label='FeatMatch', color='teal', lw=2)

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
            # 标记阶段分界线
            for bd, lbl in zip([30, 100], ['S1→S2', 'S2→S3']):
                if bd < len(history['d_audio']):
                    ax.axvline(x=bd, color='gray', ls='--', alpha=0.6, label=lbl)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'loss_ep{epoch+1}.png'), dpi=150)
        plt.close()

        # ---------- 可视化生成效果 ----------
        G_audio2acc.eval()
        G_acc2audio.eval()

        with torch.no_grad():
            vis_acc_np, vis_audio_np = next(iter(
                data_loader.get_batch_iter(4, dataset='train')
            ))
            vis_audio_list, vis_acc_list = [], []
            n_vis = min(4, vis_acc_np.shape[0])
            for b in range(n_vis):
                mel, mag = intelligent_slice_and_align(
                    vis_audio_np[b], vis_acc_np[b]
                )
                mel = to_neg1_pos1(mel)
                mag = to_neg1_pos1(mag)
                vis_audio_list.append(normalize_mel_spectrogram(mel))
                vis_acc_list.append(normalize_magnitude(mag))

            vis_audio = torch.stack(vis_audio_list).to(device)
            vis_acc   = torch.stack(vis_acc_list).to(device)

            v_fake_acc   = G_audio2acc(vis_audio)          # Audio → Acc
            v_fake_audio = G_acc2audio(vis_acc)            # Acc   → Audio
            v_rec_acc    = G_audio2acc(v_fake_audio)       # Acc   → Audio → Acc
            v_rec_audio  = G_acc2audio(v_fake_acc)         # Audio → Acc   → Audio

        G_audio2acc.train()
        G_acc2audio.train()

                # 可视化时转回 [0,1]（imshow 对 [-1,1] 显示没问题，但颜色映射更直观）
        def t2np(t):
            """[-1,1] tensor → [0,1] numpy，方便可视化"""
            arr = t.squeeze(1).cpu().numpy()
            return (arr + 1.0) * 0.5   # ← 转回 [0,1] 只为显示

#         def t2np(t):
#             return t.squeeze(1).cpu().numpy()   # [B, F, T]

        # 可视化 Acc 域
        _plot_domain(
            orig  = t2np(vis_acc),
            pred  = t2np(v_fake_acc),
            cycl  = t2np(v_rec_acc),
            title = 'Acc',
            epoch = epoch + 1,
            log_dir = log_dir,
        )

        # 可视化 Audio（Mel）域
        _plot_domain(
            orig  = t2np(vis_audio),
            pred  = t2np(v_fake_audio),
            cycl  = t2np(v_rec_audio),
            title = 'Mel',
            epoch = epoch + 1,
            log_dir = log_dir,
        )

        print(f"[Plot] Saved visualizations for epoch {epoch+1}\n")



# =====================
# 12. 保存最终模型
# =====================
final_path = os.path.join(model_dir, 'final_models.pth')
torch.save({
    'G_audio2acc_state_dict': G_audio2acc.state_dict(),
    'G_acc2audio_state_dict': G_acc2audio.state_dict(),
    'D_audio_state_dict'    : D_audio.state_dict(),
    'D_acc_state_dict'      : D_acc.state_dict(),
    'history'               : history,
}, final_path)
print(f"\nTraining completed.  Final model → {final_path}")

# import torch
# import os
# import numpy as np
# import torchaudio
# from torchsummary import summary
# import matplotlib.pyplot as plt
# from data_loader import DataLoader
# from model_drop import GeneratorAudioToAcc, GeneratorAccToAudio, DiscriminatorAudio, DiscriminatorAcc
# from loss_new import WGANLoss, generator_loss
# from tools import (
#     compute_mel_spectrogram, normalize_mel_spectrogram, compute_gradient_penalty,
#     normalize_magnitude, intelligent_slice_and_align
# )
# from torch.optim import lr_scheduler
# import torch.autograd as autograd
# import torch.nn as nn

# # =====================
# # 1. 基础配置与目录
# # =====================
# model_dir = "./models_0301.1"
# log_dir = "./logs_0301.1"
# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)

# torch.manual_seed(42)
# np.random.seed(42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # =====================
# # 2. 数据加载
# # =====================
# data_loader = DataLoader(
#     acc_path="./zrxqj_acc_sliced.npy",
#     audio_path="./zrxqj_audio_sliced.npy",
#     test_size=0.2,
#     random_state=42,
#     normalize=True,
# )

# # =====================
# # 3. 模型初始化
# # =====================
# G_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1).to(device)
# G_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1).to(device)
# D_audio = DiscriminatorAudio(input_dim=1, base=64, n_scales=1, use_spectral_norm=True).to(device)
# D_acc = DiscriminatorAcc(input_dim=1, base=64, n_scales=1, use_spectral_norm=True).to(device)

# print("Generator Audio to Acc Model Structure:")
# try:
#     summary(G_audio2acc, input_size=(1, 64, 64), device=str(device).split(':')[0])
# except Exception as e:
#     print(f"Summary error: {e}")

# # =====================
# # 4. 辅助损失函数定义
# # =====================
# class FrequencyAwareLoss(nn.Module):
#     """
#     频率感知损失：对低频区域给更高权重，惩罚高频幻觉。
#     针对acc信号主要集中在低频的特性设计。
#     """
#     def __init__(self, freq_bins=64, low_freq_ratio=0.2):
#         super().__init__()
#         weights = torch.ones(freq_bins)
#         low_end = int(freq_bins * low_freq_ratio)
#         weights[:low_end] = 3.0   # 低频区域权重×3
#         weights[low_end:] = 0.5  # 高频区域权重×0.5
#         # shape: [1, 1, freq_bins, 1] 用于广播到 [B, 1, F, T]
#         self.register_buffer('weights', weights.view(1, 1, -1, 1))

#     def forward(self, pred, target):
#         """
#         pred, target: [B, 1, F, T]
#         """
#         diff = (pred - target) ** 2
#         weighted_diff = diff * self.weights
#         return weighted_diff.mean()


# class SpectralSmoothLoss(nn.Module):
#     """
#     谱平滑损失：惩罚生成结果中频率方向的高频振荡。
#     针对predicted acc高频过多（纹理幻觉）的问题。
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         """x: [B, 1, H, W]"""
#         # H方向（频率轴）：惩罚相邻频率bin之间的剧烈变化
#         diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
#         # W方向（时间轴）：适度惩罚时间方向变化
#         diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
#         return diff_h.abs().mean() * 2.0 + diff_w.abs().mean()


# # 实例化辅助损失
# freq_loss_fn = FrequencyAwareLoss(freq_bins=64, low_freq_ratio=0.2).to(device)
# spectral_smooth = SpectralSmoothLoss().to(device)

# # =====================
# # 5. 动态权重调度函数（方案五核心）
# # =====================
# def get_dynamic_lambda(epoch):
#     """
#     三阶段动态损失权重调度：
#     - ���段1 (0~29):   以重构为主，让网络先学会基本映射关系
#     - 阶段2 (30~99):  逐渐引入并增大cycle loss
#     - 阶段3 (100+):   正常CycleGAN训练，高cycle权重
#     """
#     if epoch < 30:
#         # 阶段1：重构优先，cycle很小，对抗很弱
#         return {
#             'adv'     : 0.5,
#             'rec'     : 10.0,
#             'cycle'   : 1.0,
#             'identity': 1,
#             'freq'    : 3.0,   # 频率感知损失
#             'smooth'  : 0.1,   # 谱平滑损失
#         }
#     elif epoch < 100:
#         # 阶段2：线性过渡
#         progress = (epoch - 30) / 70.0   # 0.0 -> 1.0
#         return {
#             'adv'     : 0.5  + 0.5  * progress,   # 0.1  -> 1.0
#             'rec'     : 10.0 - 5.0  * progress,   # 10.0 -> 5.0
#             'cycle'   : 1.0  + 7.0  * progress,   # 1.0  -> 8.0
#             'identity': 1  - 0.5  * progress,   # 2.0  -> 1.0
#             'freq'    : 3.0  - 1.0  * progress,   # 3.0  -> 2.0
#             'smooth'  : 0.1,
#         }
#     else:
#         # 阶段3：正常训练
#         return {
#             'adv'     : 1.0,
#             'rec'     : 5.0,
#             'cycle'   : 8.0,
#             'identity': 0.5,
#             'freq'    : 2.0,
#             'smooth'  : 0.1,
#         }


# # =====================
# # 6. 优化器与调度器
# # =====================
# lr_G  = 1e-4
# lr_D  = 2e-4
# beta1, beta2 = 0.5, 0.999

# optimizer_G = torch.optim.Adam(
#     list(G_audio2acc.parameters()) + list(G_acc2audio.parameters()),
#     lr=lr_G, betas=(beta1, beta2)
# )
# optimizer_D_audio = torch.optim.Adam(D_audio.parameters(), lr=lr_D, betas=(beta1, beta2))
# optimizer_D_acc   = torch.optim.Adam(D_acc.parameters(),   lr=lr_D, betas=(beta1, beta2))

# # WGAN-GP 梯度惩罚系数
# lambda_gp = 10.0

# # 训练周期配置
# num_epochs      = 500
# batch_size      = 16
# warmup, N0, N1  = 50, 350, 150


# def lr_lambda(epoch):
#     if epoch < warmup:
#         return (epoch + 1) / warmup
#     e = epoch - warmup
#     if e < N0:
#         return 1.0
#     t = (e - N0) / float(N1)
#     return max(0.0, 1.0 - t)


# scheduler_G       = lr_scheduler.LambdaLR(optimizer_G,       lr_lambda=lr_lambda)
# scheduler_D_audio = lr_scheduler.LambdaLR(optimizer_D_audio, lr_lambda=lr_lambda)
# scheduler_D_acc   = lr_scheduler.LambdaLR(optimizer_D_acc,   lr_lambda=lr_lambda)

# wgan_loss = WGANLoss()

# # =====================
# # 7. 损失记录容器
# # =====================
# d_audio_losses      = []
# d_acc_losses        = []
# g_audio2acc_losses  = []
# g_acc2audio_losses  = []
# cycle_losses        = []
# freq_cycle_losses   = []
# smooth_losses       = []

# # 辅助：当前epoch各阶段的权重（用于日志）
# current_lambdas = {}

# # =====================
# # 8. 训练循环
# # =====================
# print("Starting Training Loop...")

# total_epochs = warmup + N0 + N1

# for epoch in range(total_epochs):

#     # ---------- 获取当前epoch的动态权重 ----------
#     current_lambdas = get_dynamic_lambda(epoch)
#     lambda_adv      = current_lambdas['adv']
#     lambda_rec      = current_lambdas['rec']
#     lambda_cycle    = current_lambdas['cycle']
#     lambda_identity = current_lambdas['identity']
#     lambda_freq     = current_lambdas['freq']
#     lambda_smooth   = current_lambdas['smooth']

#     # 每个阶段切换时打印权重
#     if epoch in (0, 30, 100):
#         print(f"\n[Epoch {epoch}] Lambda weights updated:")
#         for k, v in current_lambdas.items():
#             print(f"  {k:10s}: {v:.4f}")
#         print()

#     # ---------- Epoch级别累计量 ----------
#     total_d_audio_loss     = 0.0
#     total_d_acc_loss       = 0.0
#     total_g_audio2acc_loss = 0.0
#     total_g_acc2audio_loss = 0.0
#     total_cycle_loss       = 0.0
#     total_freq_cycle_loss  = 0.0
#     total_smooth_loss      = 0.0
#     num_batches            = 0

#     # Critic训练次数：早期多训判别器
#     critic_iterations = 5 if epoch < 50 else 2

#     # ---------- Batch迭代 ----------
#     for i, (acc_batch_np, audio_batch_np) in enumerate(
#             data_loader.get_batch_iter(batch_size, dataset='train')):

#         num_batches += 1
#         current_bs = acc_batch_np.shape[0]

#         # -------------------------------------------------
#         # A. 数据预处理：逐样本切片、特征提取、归一化
#         # -------------------------------------------------
#         proc_audio_list = []
#         proc_acc_list   = []

#         for b in range(current_bs):
#             a_raw = audio_batch_np[b]
#             c_raw = acc_batch_np[b]

#             mel, mag = intelligent_slice_and_align(a_raw, c_raw)

#             mel = normalize_mel_spectrogram(mel)
#             mag = normalize_magnitude(mag)

#             proc_audio_list.append(mel)
#             proc_acc_list.append(mag)

#         # Shape: [B, 1, 128, 128]
#         audio_mel = torch.stack(proc_audio_list).to(device)
#         magnitude = torch.stack(proc_acc_list).to(device)

#         # -------------------------------------------------
#         # B. 判别器训练（WGAN-GP）
#         # -------------------------------------------------
#         for _ in range(critic_iterations):

#             # --- D_Audio (Acc -> Audio) ---
#             optimizer_D_audio.zero_grad()

#             fake_audio_mel       = G_acc2audio(magnitude)
#             real_audio_validity  = D_audio(audio_mel)
#             fake_audio_validity  = D_audio(fake_audio_mel.detach())

#             d_audio_wgan = wgan_loss.discriminator_loss(real_audio_validity, fake_audio_validity)
#             gp_audio     = compute_gradient_penalty(D_audio, audio_mel, fake_audio_mel, device)
#             d_audio_loss = d_audio_wgan + lambda_gp * gp_audio

#             d_audio_loss.backward()
#             optimizer_D_audio.step()

#             # --- D_Acc (Audio -> Acc) ---
#             optimizer_D_acc.zero_grad()

#             fake_acc_mag       = G_audio2acc(audio_mel)
#             real_acc_validity  = D_acc(magnitude)
#             fake_acc_validity  = D_acc(fake_acc_mag.detach())

#             d_acc_wgan = wgan_loss.discriminator_loss(real_acc_validity, fake_acc_validity)
#             gp_acc     = compute_gradient_penalty(D_acc, magnitude, fake_acc_mag, device)
#             d_acc_loss = d_acc_wgan + lambda_gp * gp_acc

#             d_acc_loss.backward()
#             optimizer_D_acc.step()

#         total_d_audio_loss += d_audio_loss.item()
#         total_d_acc_loss   += d_acc_loss.item()

#         # -------------------------------------------------
#         # C. 生成器训练
#         # -------------------------------------------------
#         optimizer_G.zero_grad()

#         # ---------- C-1. 前向传播 ----------
#         fake_audio_mel      = G_acc2audio(magnitude)      # Acc  -> Fake Audio
#         fake_audio_validity = D_audio(fake_audio_mel)

#         fake_acc_mag        = G_audio2acc(audio_mel)      # Audio -> Fake Acc
#         fake_acc_validity   = D_acc(fake_acc_mag)

#         # ---------- C-2. 循环一致性 ----------
#         recovered_acc_mag   = G_audio2acc(fake_audio_mel) # Acc  -> Audio -> Acc
#         recovered_audio_mel = G_acc2audio(fake_acc_mag)   # Audio -> Acc  -> Audio

#         # ---------- C-3. 身份映射 ----------
#         identity_audio_mel = G_acc2audio(audio_mel) if lambda_identity > 0 else None
#         identity_acc_mag   = G_audio2acc(magnitude)  if lambda_identity > 0 else None

#         # ---------- C-4. 基础CycleGAN损失 ----------
#         # Audio->Acc 方向（使用当前epoch动态权重）
#         g_audio2acc_total, _, _, g_audio2acc_cyc, _ = generator_loss(
#             fake_acc_validity, magnitude, fake_acc_mag, recovered_acc_mag,
#             identity_acc_mag,
#             lambda_adv, lambda_rec, lambda_cycle, lambda_identity
#         )

#         # Acc->Audio 方向
#         g_acc2audio_total, _, _, g_acc2audio_cyc, _ = generator_loss(
#             fake_audio_validity, audio_mel, fake_audio_mel, recovered_audio_mel,
#             identity_audio_mel,
#             lambda_adv, lambda_rec, lambda_cycle, lambda_identity
#         )

#         # ---------- C-5. 频率感知循环一致性损失（方案二）----------
#         # 重点惩罚 Acc域 的低频重建误差（recovered_acc_mag应与magnitude一致）
#         freq_cycle_acc   = freq_loss_fn(recovered_acc_mag,   magnitude)
#         freq_cycle_audio = freq_loss_fn(recovered_audio_mel, audio_mel)
#         freq_cycle_loss  = freq_cycle_acc + freq_cycle_audio

#         # ---------- C-6. 谱平滑损失（方案三）----------
#         # 只对acc域的生成结果施加平滑约束，抑制高频幻觉
#         smooth_loss_fake = spectral_smooth(fake_acc_mag)
#         smooth_loss_rec  = spectral_smooth(recovered_acc_mag)
#         smooth_loss      = smooth_loss_fake + smooth_loss_rec

#         # ---------- C-7. 汇总损失 ----------
#         g_total    = (g_audio2acc_total
#                     + g_acc2audio_total
#                     + lambda_freq   * freq_cycle_loss
#                     + lambda_smooth * smooth_loss)

#         cycle_loss = g_audio2acc_cyc + g_acc2audio_cyc

#         g_total.backward()

#         # 梯度裁剪，防止训练不稳定
#         torch.nn.utils.clip_grad_norm_(
#             list(G_audio2acc.parameters()) + list(G_acc2audio.parameters()),
#             max_norm=1.0
#         )

#         optimizer_G.step()

#         # ---------- C-8. 累计记录 ----------
#         total_g_audio2acc_loss += g_audio2acc_total.item()
#         total_g_acc2audio_loss += g_acc2audio_total.item()
#         total_cycle_loss       += cycle_loss.item()
#         total_freq_cycle_loss  += freq_cycle_loss.item()
#         total_smooth_loss      += smooth_loss.item()

#         # ---------- C-9. Batch级日志 ----------
#         if i % 50 == 0:
#             print(
#                 f"[Epoch {epoch:>3d}/{total_epochs} | Batch {i:>4d}] "
#                 f"Stage: {'1-Reconstruct' if epoch < 30 else ('2-Transition' if epoch < 100 else '3-CycleGAN')} | "
#                 f"D_audio: {d_audio_loss.item():>7.4f}  D_acc: {d_acc_loss.item():>7.4f} | "
#                 f"G_total: {g_total.item():>7.4f}  Cycle: {cycle_loss.item():>7.4f} | "
#                 f"FreqCyc: {freq_cycle_loss.item():>7.4f}  Smooth: {smooth_loss.item():>7.4f}"
#             )

#     # =====================
#     # 9. Epoch结束处理
#     # =====================

#     # ---------- 更新学习率调度器 ----------
#     scheduler_G.step()
#     scheduler_D_audio.step()
#     scheduler_D_acc.step()

#     # ---------- 计算Epoch平均损失 ----------
#     avg_d_audio     = total_d_audio_loss     / num_batches
#     avg_d_acc       = total_d_acc_loss       / num_batches
#     avg_g_audio2acc = total_g_audio2acc_loss / num_batches
#     avg_g_acc2audio = total_g_acc2audio_loss / num_batches
#     avg_cycle       = total_cycle_loss       / num_batches
#     avg_freq_cycle  = total_freq_cycle_loss  / num_batches
#     avg_smooth      = total_smooth_loss      / num_batches

#     # ---------- 追加到记录列表 ----------
#     d_audio_losses.append(avg_d_audio)
#     d_acc_losses.append(avg_d_acc)
#     g_audio2acc_losses.append(avg_g_audio2acc)
#     g_acc2audio_losses.append(avg_g_acc2audio)
#     cycle_losses.append(avg_cycle)
#     freq_cycle_losses.append(avg_freq_cycle)
#     smooth_losses.append(avg_smooth)

#     # ---------- Epoch级日志 ----------
#     current_lr_G = optimizer_G.param_groups[0]['lr']
#     print(
#         f"\n>>> Epoch {epoch + 1:>3d}/{total_epochs} Summary | "
#         f"LR_G: {current_lr_G:.2e} | "
#         f"λ_adv={lambda_adv:.1f} λ_rec={lambda_rec:.1f} "
#         f"λ_cyc={lambda_cycle:.1f} λ_id={lambda_identity:.1f} "
#         f"λ_freq={lambda_freq:.1f} λ_smooth={lambda_smooth:.2f}\n"
#         f"    D_audio: {avg_d_audio:.4f}  D_acc: {avg_d_acc:.4f}\n"
#         f"    G_a2acc: {avg_g_audio2acc:.4f}  G_acc2a: {avg_g_acc2audio:.4f}\n"
#         f"    Cycle:   {avg_cycle:.4f}  FreqCyc: {avg_freq_cycle:.4f}  Smooth: {avg_smooth:.4f}\n"
#     )

#     # =====================
#     # 10. 定期保存与可视化
#     # =====================
#     if (epoch + 1) % 50 == 0:

#         # ---------- 保存权重 ----------
#         save_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch + 1}.pth')
#         torch.save({
#             'epoch'                  : epoch + 1,
#             'G_audio2acc_state_dict' : G_audio2acc.state_dict(),
#             'G_acc2audio_state_dict' : G_acc2audio.state_dict(),
#             'D_audio_state_dict'     : D_audio.state_dict(),
#             'D_acc_state_dict'       : D_acc.state_dict(),
#             'optimizer_G_state_dict' : optimizer_G.state_dict(),
#             'optimizer_D_audio_state_dict': optimizer_D_audio.state_dict(),
#             'optimizer_D_acc_state_dict'  : optimizer_D_acc.state_dict(),
#             'scheduler_G_state_dict' : scheduler_G.state_dict(),
#             'd_audio_losses'         : d_audio_losses,
#             'd_acc_losses'           : d_acc_losses,
#             'g_audio2acc_losses'     : g_audio2acc_losses,
#             'g_acc2audio_losses'     : g_acc2audio_losses,
#             'cycle_losses'           : cycle_losses,
#             'freq_cycle_losses'      : freq_cycle_losses,
#             'smooth_losses'          : smooth_losses,
#             'current_lambdas'        : current_lambdas,
#         }, save_path)
#         print(f"[Checkpoint] Saved to {save_path}")

#         # ---------- 绘制Loss曲线 ----------
#         fig, axes = plt.subplots(3, 1, figsize=(12, 12))

#         # 子图1: 判别器损失
#         axes[0].plot(d_audio_losses, label='D_Audio', color='steelblue')
#         axes[0].plot(d_acc_losses,   label='D_Acc',   color='darkorange')
#         axes[0].set_title('Discriminator Losses')
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('Loss')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
#         # 标记阶段分界线
#         for boundary, label in zip([30, 100], ['Stage1→2', 'Stage2→3']):
#             if boundary < len(d_audio_losses):
#                 axes[0].axvline(x=boundary, color='red', linestyle='--', alpha=0.5, label=label)

#         # 子图2: 生成器损失 + Cycle损失
#         axes[1].plot(g_audio2acc_losses, label='G_audio2acc', color='green')
#         axes[1].plot(g_acc2audio_losses, label='G_acc2audio', color='purple')
#         axes[1].plot(cycle_losses,       label='Cycle Loss',  color='red',   linewidth=2)
#         axes[1].set_title('Generator & Cycle Losses')
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('Loss')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
#         for boundary in [30, 100]:
#             if boundary < len(g_audio2acc_losses):
#                 axes[1].axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

#         # 子图3: 辅助损失（频率感知 + 谱平滑）
#         axes[2].plot(freq_cycle_losses, label='Freq Cycle Loss', color='teal')
#         axes[2].plot(smooth_losses,     label='Smooth Loss',     color='brown')
#         axes[2].set_title('Auxiliary Losses (FreqCycle + Smooth)')
#         axes[2].set_xlabel('Epoch')
#         axes[2].set_ylabel('Loss')
#         axes[2].legend()
#         axes[2].grid(True, alpha=0.3)
#         for boundary in [30, 100]:
#             if boundary < len(freq_cycle_losses):
#                 axes[2].axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

#         plt.tight_layout()
#         fig_path = os.path.join(log_dir, f'loss_epoch_{epoch + 1}.png')
#         plt.savefig(fig_path, dpi=150)
#         plt.close()
#         print(f"[Plot] Loss curve saved to {fig_path}")

#         # ---------- 可视化生成效果（固定样本）----------
#         G_audio2acc.eval()
#         G_acc2audio.eval()
#         with torch.no_grad():
#             # 取前两个batch样本做可视化
#             vis_acc_np, vis_audio_np = next(iter(
#                 data_loader.get_batch_iter(2, dataset='train')
#             ))
#             vis_audio_list, vis_acc_list = [], []
#             for b in range(min(2, vis_acc_np.shape[0])):
#                 mel, mag = intelligent_slice_and_align(vis_audio_np[b], vis_acc_np[b])
#                 mel = normalize_mel_spectrogram(mel)
#                 mag = normalize_magnitude(mag)
#                 vis_audio_list.append(mel)
#                 vis_acc_list.append(mag)

#             vis_audio = torch.stack(vis_audio_list).to(device)  # [2,1,128,128]
#             vis_acc   = torch.stack(vis_acc_list).to(device)

#             # 前向推理
#             vis_fake_acc        = G_audio2acc(vis_audio)         # Audio -> Acc
#             vis_fake_audio      = G_acc2audio(vis_acc)           # Acc   -> Audio
#             vis_recovered_acc   = G_audio2acc(vis_fake_audio)    # Acc   -> Audio -> Acc
#             vis_recovered_audio = G_acc2audio(vis_fake_acc)      # Audio -> Acc   -> Audio

#             def to_numpy(t):
#                 return t.squeeze(1).cpu().numpy()   # [B, H, W]

#             orig_acc   = to_numpy(vis_acc)
#             pred_acc   = to_numpy(vis_fake_acc)
#             cycl_acc   = to_numpy(vis_recovered_acc)
#             orig_audio = to_numpy(vis_audio)
#             pred_audio = to_numpy(vis_fake_audio)
#             cycl_audio = to_numpy(vis_recovered_audio)

#         G_audio2acc.train()
#         G_acc2audio.train()

#         # 绘制 Acc 域对比（原始 / 预测 / 循环恢复）
#         fig, axes = plt.subplots(3, 2, figsize=(14, 10))
#         row_labels = ['Original Acc', 'Predicted Acc', 'Cycled Acc']
#         for col in range(2):
#             for row, (data, label) in enumerate(zip(
#                 [orig_acc, pred_acc, cycl_acc], row_labels
#             )):
#                 im = axes[row, col].imshow(
#                     data[col], aspect='auto', origin='lower',
#                     cmap='viridis',
#                     vmin=data[col].min(), vmax=data[col].max()
#                 )
#                 axes[row, col].set_title(f'{label} (Sample {col})')
#                 axes[row, col].set_xlabel('Time Frame')
#                 axes[row, col].set_ylabel('Freq Bin')
#                 plt.colorbar(im, ax=axes[row, col])

#         plt.suptitle(
#             f'Acc Domain Visualization @ Epoch {epoch + 1} | '
#             f'Stage: {"1" if epoch < 30 else ("2" if epoch < 100 else "3")}',
#             fontsize=13
#         )
#         plt.tight_layout()
#         vis_path = os.path.join(log_dir, f'vis_acc_epoch_{epoch + 1}.png')
#         plt.savefig(vis_path, dpi=150)
#         plt.close()
#         print(f"[Plot] Acc visualization saved to {vis_path}\n")

# # =====================
# # 11. 保存最终模型
# # =====================
# final_path = os.path.join(model_dir, 'final_models.pth')
# torch.save({
#     'G_audio2acc_state_dict': G_audio2acc.state_dict(),
#     'G_acc2audio_state_dict': G_acc2audio.state_dict(),
#     'D_audio_state_dict'    : D_audio.state_dict(),
#     'D_acc_state_dict'      : D_acc.state_dict(),
#     'd_audio_losses'        : d_audio_losses,
#     'd_acc_losses'          : d_acc_losses,
#     'g_audio2acc_losses'    : g_audio2acc_losses,
#     'g_acc2audio_losses'    : g_acc2audio_losses,
#     'cycle_losses'          : cycle_losses,
#     'freq_cycle_losses'     : freq_cycle_losses,
#     'smooth_losses'         : smooth_losses,
# }, final_path)

# print(f"\nTraining completed. Final model saved to {final_path}")
