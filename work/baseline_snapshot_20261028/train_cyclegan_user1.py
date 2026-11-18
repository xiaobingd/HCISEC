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
from tools import compute_gradient_penalty
from torch.optim import lr_scheduler

# =====================
# 1. 基础配置
# =====================
model_dir = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/model"
log_dir   = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir,   exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# 2. 数据加载
# =====================
# 数据已经是预处理好的频谱图 (N, 80, 80)，归一化到 [-1, 1]
# ACC: STFT频谱图, Audio: Mel频谱图
DATA_DIR = "./"

data_loader = DataLoader(
    acc_path   = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/data_split/user1/cyclegan_acc.npy",
    audio_path = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/data_split/user1/cyclegan_audio.npy",
    test_size    = 0.1,
    val_size     = 0.1,
    random_state = 42,
)

# =====================
# 3. 模型初始化
# =====================
# 输入尺寸: (1, 80, 80) — 80频率bins × 80时间帧
G_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1).to(device)
G_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1).to(device)
D_audio     = DiscriminatorAudio(input_dim=1, base=64, n_scales=1,
                                  use_spectral_norm=True).to(device)
D_acc       = DiscriminatorAcc(input_dim=1,   base=64, n_scales=1,
                                use_spectral_norm=True).to(device)

try:
    summary(G_audio2acc, input_size=(1, 80, 80),
            device=str(device).split(':')[0])
except Exception as e:
    print(f"Summary error: {e}")

# =====================
# 4. 辅助损失函数
# =====================
class FrequencyAwareLoss(nn.Module):
    """
    频率加权MSE，根据模态特性使用不同权重分布。
    - Mel (Audio)：低频能量集中，低频加权
    - STFT (Acc) ：谐波分布在全频段，均匀加权
    """
    def __init__(self, low_freq_ratio=0.2, mode='mel'):
        super().__init__()
        self.low_freq_ratio   = low_freq_ratio
        self.mode             = mode
        self._cached_weight   = None
        self._cached_freq_bins = None

    def _get_weights(self, freq_bins, device):
        if self._cached_freq_bins != freq_bins:
            w = torch.ones(freq_bins)
            if self.mode == 'mel':
                # Mel域：低频能量密集更重要
                low_end = int(freq_bins * self.low_freq_ratio)
                w[:low_end] = 1.5
                w[low_end:] = 0.5
            else:
                # STFT域：谐波分布在全频段，均匀权重
                w[:] = 1.0
            self._cached_weight    = w.view(1, 1, -1, 1).to(device)
            self._cached_freq_bins = freq_bins
        return self._cached_weight

    def forward(self, pred, target):
        freq_bins = pred.shape[2]
        weights   = self._get_weights(freq_bins, pred.device)
        diff      = (pred - target) ** 2
        return (diff * weights).mean()


class HarmonicPreserveLoss(nn.Module):
    """
    Acc (STFT) 域专用：保留谐波峰结构。
    不惩罚绝对梯度大小，而是惩罚 pred 与 target 频率梯度模式的偏差。
    让生成的 acc 频谱在该尖锐的地方尖锐，该平坦的地方平坦。
    """
    def forward(self, pred, target):
        # 频率轴梯度模式匹配
        pred_dh   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
        target_dh = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_freq = F.l1_loss(pred_dh, target_dh)

        # 时间轴梯度模式匹配
        pred_dw   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
        target_dw = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_time = F.l1_loss(pred_dw, target_dw)

        return grad_freq + grad_time * 0.5


class TemporalCoherenceLoss(nn.Module):
    """
    Audio (Mel) 域专用：保持时间轴连续性。
    Mel频谱在时间轴上应该平滑过渡，惩罚生成结果中的时间跳变。
    """
    def forward(self, x):
        diff_t = x[:, :, :, 1:] - x[:, :, :, :-1]   # 时间轴
        return diff_t.abs().mean()


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


# Acc (STFT) 域：全频段均匀权重
freq_loss_acc    = FrequencyAwareLoss(mode='stft').to(device)
# Audio (Mel) 域：低频加权
freq_loss_mel    = FrequencyAwareLoss(low_freq_ratio=0.2, mode='mel').to(device)
# Acc 域：谐波结构保持
harmonic_loss_fn = HarmonicPreserveLoss().to(device)
# Audio 域：时间连续性
temporal_loss_fn = TemporalCoherenceLoss().to(device)
feat_match_loss  = FeatureMatchingLoss().to(device)

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
                vmin=-1, vmax=1
            )
            axes[row, col].set_title(f'{lbl} (Sample {col})', fontsize=10)
            axes[row, col].set_xlabel('Time Frame')
            axes[row, col].set_ylabel('Freq Bin')
            plt.colorbar(im, ax=axes[row, col])

    stage = '1' if epoch <= 50 else ('2' if epoch <= 150 else '3')
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
def get_dynamic_lambda(epoch):
    """
    Stage1 (0-49)   : 重构优先，网络先学会基本映射
    Stage2 (50-149) : 线性过渡，逐步引入 cycle / feat_match
    Stage3 (150+)   : 正常 CycleGAN，feat_match 主导清晰度
    """
    if epoch < 50:
        return dict(adv=0.5, rec=10.0, cycle=1.0, identity=1.0,
                    freq=2.0, harmonic=1.0, temporal=0.5, feat_match=2.0)
    elif epoch < 150:
        p = (epoch - 50) / 100.0          # 0→1
        return dict(
            adv        = 0.5  + 0.5  * p,
            rec        = 10.0 - 3.0  * p,
            cycle      = 1.0  + 7.0  * p,
            identity   = 1.0  - 0.9  * p,
            freq       = 2.0,
            harmonic   = 1.0  + 1.0  * p,
            temporal   = 0.5  + 0.5  * p,
            feat_match = 2.0  + 2.0  * p,
        )
    else:
        return dict(adv=1.0, rec=7.0, cycle=8.0, identity=0.1,
                    freq=2.0, harmonic=2.0, temporal=1.0, feat_match=4.0)

# =====================
# 6. 优化器与调度器
# =====================
lr_G  = 1e-4
lr_D  = 1e-4
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
total_epochs           = 200

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
    harmonic=[], temporal=[], feat_match=[]
)
current_lambdas = {}

# =====================
# 8. 训练循环
# =====================
print("Starting Training Loop...")
print(f"Data: ACC & Audio spectrograms, shape (80, 80), normalized to [-1, 1]")
print(f"Total epochs: {total_epochs}, batch_size: {batch_size}")

for epoch in range(total_epochs):

    # ---- 获取本 epoch 动态权重 ----
    current_lambdas  = get_dynamic_lambda(epoch)
    lambda_adv       = current_lambdas['adv']
    lambda_rec       = current_lambdas['rec']
    lambda_cycle     = current_lambdas['cycle']
    lambda_identity  = current_lambdas['identity']
    lambda_freq      = current_lambdas['freq']
    lambda_harmonic  = current_lambdas['harmonic']
    lambda_temporal  = current_lambdas['temporal']
    lambda_feat      = current_lambdas['feat_match']

    if epoch in (0, 50, 150):
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Lambda weights →  {current_lambdas}")
        print(f"{'='*60}\n")

    # ---- Epoch 累计量 ----
    ep = {k: 0.0 for k in
          ['d_audio','d_acc','g_a2acc','g_acc2a',
           'cycle','freq_cycle','harmonic','temporal','feat_match']}
    num_batches       = 0
    critic_iterations = 5 if epoch < 50 else 2

    for i, (acc_np, audio_np) in enumerate(
            data_loader.get_batch_iter(batch_size, dataset='train')):

        num_batches  += 1
        current_bs    = acc_np.shape[0]

        # -------------------------------------------------
        # A. 数据准备 — 直接转为tensor，数据已经是(B,80,80)归一化到[-1,1]
        # -------------------------------------------------
        # 添加channel维度: (B, 80, 80) -> (B, 1, 80, 80)
        
        audio_mel= torch.from_numpy(audio_np).float().unsqueeze(1).to(device)
        audio_mel =audio_mel[:,:,:80,:80]
        
        magnitude= torch.from_numpy(acc_np).float().unsqueeze(1).to(device)
        magnitude=magnitude[:,:,:80,:80]
        
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
        feat_audio = feat_match_loss(real_audio_feats, fake_audio_feats)
        feat_acc   = feat_match_loss(real_acc_feats,   fake_acc_feats)
        feat_total = feat_audio + feat_acc

        # C-7. 频率感知循环损失（各域使用各自的频率权重）
        #   Acc (STFT)：全频段均匀，谐波分布广
        #   Audio (Mel)：低频加权，能量集中在低频
        freq_cycle_loss_acc   = freq_loss_acc(recovered_acc_mag, magnitude)
        freq_cycle_loss_audio = freq_loss_mel(recovered_audio_mel, audio_mel)
        freq_cycle_loss = freq_cycle_loss_acc + freq_cycle_loss_audio

        # C-8. 模态特异性损失
        #   Acc 域：谐波结构保持（梯度模式匹配，不惩罚尖锐本身）
        harmonic_loss = (harmonic_loss_fn(fake_acc_mag, magnitude)
                       + harmonic_loss_fn(recovered_acc_mag, magnitude))
        #   Audio 域：时间连续性（Mel 频谱时间轴应平滑过渡）
        temporal_loss = (temporal_loss_fn(fake_audio_mel)
                       + temporal_loss_fn(recovered_audio_mel))

        # C-9. 汇总
        g_total    = (g_a2acc_total
                    + g_acc2a_total
                    + lambda_feat     * feat_total
                    + lambda_freq     * freq_cycle_loss
                    + lambda_harmonic * harmonic_loss
                    + lambda_temporal * temporal_loss)
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
        ep['harmonic']     += harmonic_loss.item()
        ep['temporal']     += temporal_loss.item()
        ep['feat_match']   += feat_total.item()

        # C-11. Batch 日志
        if i % 5 == 0:
            stage = ('1-Rec' if epoch < 50
                     else ('2-Trans' if epoch < 150 else '3-Cycle'))
            print(
                f"[Ep {epoch:>3d}/{total_epochs} | Bt {i:>4d} | {stage}] "
                f"D: {d_audio_loss.item():.3f}/{d_acc_loss.item():.3f} | "
                f"G: {g_total.item():.3f} "
                f"Cyc: {cycle_loss.item():.3f} | "
                f"Feat: {feat_total.item():.3f} "
                f"(aud={feat_audio.item():.3f} acc={feat_acc.item():.3f}) | "
                f"Freq: {freq_cycle_loss.item():.3f} "
                f"Harm: {harmonic_loss.item():.3f} "
                f"Temp: {temporal_loss.item():.3f}"
            )

    # =====================
    # 9. Epoch 结束
    # =====================
    scheduler_G.step()
    scheduler_D_audio.step()
    scheduler_D_acc.step()

    # 计算平均并记录
    for k in ep:
        history[k].append(ep[k] / max(num_batches, 1))

    lr_now = optimizer_G.param_groups[0]['lr']
    print(
        f"\n>>> Ep {epoch+1:>3d}/{total_epochs} | LR={lr_now:.2e} | "
        f"λ(adv={lambda_adv:.1f} rec={lambda_rec:.1f} "
        f"cyc={lambda_cycle:.1f} id={lambda_identity:.1f} "
        f"freq={lambda_freq:.1f} harm={lambda_harmonic:.1f} "
        f"temp={lambda_temporal:.1f} feat={lambda_feat:.1f})\n"
        f"    D_audio={history['d_audio'][-1]:.4f}  "
        f"D_acc={history['d_acc'][-1]:.4f}\n"
        f"    G_a2acc={history['g_a2acc'][-1]:.4f}  "
        f"G_acc2a={history['g_acc2a'][-1]:.4f}\n"
        f"    Cycle={history['cycle'][-1]:.4f}  "
        f"FreqCyc={history['freq_cycle'][-1]:.4f}  "
        f"Harmonic={history['harmonic'][-1]:.4f}  "
        f"Temporal={history['temporal'][-1]:.4f}  "
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

        axes[2].set_title('Cycle & Frequency Loss')
        axes[2].plot(history['cycle'],      label='Cycle',    color='red',    lw=2)
        axes[2].plot(history['freq_cycle'], label='FreqCycle',color='tomato', lw=1.5)

        axes[3].set_title('Modality-Specific Loss')
        axes[3].plot(history['harmonic'],   label='Harmonic(Acc)',  color='teal',   lw=2)
        axes[3].plot(history['temporal'],   label='Temporal(Audio)',color='coral',   lw=2)
        axes[3].plot(history['feat_match'], label='FeatMatch',      color='slategray', lw=1.5)

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
            for bd, lbl in zip([50, 150], ['S1→S2', 'S2→S3']):
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
                data_loader.get_batch_iter(4, dataset='val', shuffle=False)
            ))
            n_vis = min(4, vis_acc_np.shape[0])
            
            # 直接转tensor，添加channel维度
            vis_audio = torch.from_numpy(vis_audio_np[:n_vis]).float().unsqueeze(1).to(device)
            vis_acc   = torch.from_numpy(vis_acc_np[:n_vis]).float().unsqueeze(1).to(device)

            v_fake_acc   = G_audio2acc(vis_audio)          # Audio → Acc
            v_fake_audio = G_acc2audio(vis_acc)            # Acc   → Audio
            v_rec_acc    = G_audio2acc(v_fake_audio)       # Acc   → Audio → Acc
            v_rec_audio  = G_acc2audio(v_fake_acc)         # Audio → Acc   → Audio

        G_audio2acc.train()
        G_acc2audio.train()

        def t2np(t):
            return t.squeeze(1).cpu().numpy()   # [B, F, T]

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
# from torchsummary import summary
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F

# from data_loader import DataLoader
# from model_drop import (GeneratorAudioToAcc, GeneratorAccToAudio,
#                         DiscriminatorAudio, DiscriminatorAcc)
# from loss_new import WGANLoss, generator_loss
# from tools import compute_gradient_penalty
# from torch.optim import lr_scheduler

# # =====================
# # 1. 基础配置
# # =====================
# model_dir = "./models_bo150y20"
# log_dir   = "./logs_bo150y20"
# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(log_dir,   exist_ok=True)

# torch.manual_seed(42)
# np.random.seed(42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # =====================
# # 2. 数据加载
# # =====================
# # 数据已经是预处理好的频谱图 (N, 80, 80)，归一化到 [-1, 1]
# # ACC: STFT频谱图, Audio: Mel频谱图
# DATA_DIR = "./"

# data_loader = DataLoader(
#     acc_path   = "/root/autodl-tmp/cycle1/meiy/3/acc.npy",
#     audio_path = "/root/autodl-tmp/cycle1/meiy/3/audio.npy",
#     test_size    = 0.1,
#     val_size     = 0.1,
#     random_state = 42,
# )

# # =====================
# # 3. 模型初始化
# # =====================
# # 输入尺寸: (1, 80, 80) — 80频率bins × 80时间帧
# G_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1).to(device)
# G_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1).to(device)
# D_audio     = DiscriminatorAudio(input_dim=1, base=64, n_scales=1,
#                                   use_spectral_norm=True).to(device)
# D_acc       = DiscriminatorAcc(input_dim=1,   base=64, n_scales=1,
#                                 use_spectral_norm=True).to(device)

# try:
#     summary(G_audio2acc, input_size=(1, 80, 80),
#             device=str(device).split(':')[0])
# except Exception as e:
#     print(f"Summary error: {e}")

# # =====================
# # 4. 辅助损失函数
# # =====================
# class FrequencyAwareLoss(nn.Module):
#     """低频加权MSE，自动适配输入频率维度"""
#     def __init__(self, low_freq_ratio=0.2):
#         super().__init__()
#         self.low_freq_ratio   = low_freq_ratio
#         self._cached_weight   = None
#         self._cached_freq_bins = None

#     def _get_weights(self, freq_bins, device):
#         if self._cached_freq_bins != freq_bins:
#             w = torch.ones(freq_bins)
#             low_end = int(freq_bins * self.low_freq_ratio)
#             w[:low_end] = 1.0
#             w[low_end:] = 0.5
#             self._cached_weight    = w.view(1, 1, -1, 1).to(device)
#             self._cached_freq_bins = freq_bins
#         return self._cached_weight

#     def forward(self, pred, target):
#         freq_bins = pred.shape[2]
#         weights   = self._get_weights(freq_bins, pred.device)
#         diff      = (pred - target) ** 2
#         return (diff * weights).mean()


# class SpectralSmoothLoss(nn.Module):
#     """惩罚频率方向高频振荡，只用于 acc 域"""
#     def forward(self, x):
#         diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]   # 频率轴
#         diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]   # 时间轴
#         return diff_h.abs().mean() * 2.0 + diff_w.abs().mean()


# class FeatureMatchingLoss(nn.Module):
#     """
#     从判别器中间层提取特征计算L1距离。
#     比像素级L2更能保留谐波等高频结构细节。
#     """
#     def forward(self, real_feats, fake_feats):
#         loss = 0.0
#         for rf, ff in zip(real_feats, fake_feats):
#             loss += F.l1_loss(ff, rf.detach())
#         return loss / max(len(real_feats), 1)


# freq_loss_fn    = FrequencyAwareLoss(low_freq_ratio=0.2).to(device)
# spectral_smooth = SpectralSmoothLoss().to(device)
# feat_match_loss = FeatureMatchingLoss().to(device)

# # =====================
# # 11. 可视化辅助函数
# # =====================
# def _plot_domain(orig, pred, cycl, title, epoch, log_dir):
#     """
#     orig / pred / cycl : numpy [B, F, T]
#     绘制前 min(B,2) 个样本的 原始/预测/循环恢复 对比图
#     """
#     n = min(orig.shape[0], 2)
#     fig, axes = plt.subplots(3, n, figsize=(7 * n, 12))
#     if n == 1:
#         axes = axes[:, np.newaxis]

#     row_labels = [f'Original {title}', f'Predicted {title}', f'Cycled {title}']
#     data_rows  = [orig, pred, cycl]

#     for row, (data, lbl) in enumerate(zip(data_rows, row_labels)):
#         for col in range(n):
#             im = axes[row, col].imshow(
#                 data[col], aspect='auto', origin='lower',
#                 cmap='viridis',
#                 vmin=-1, vmax=1
#             )
#             axes[row, col].set_title(f'{lbl} (Sample {col})', fontsize=10)
#             axes[row, col].set_xlabel('Time Frame')
#             axes[row, col].set_ylabel('Freq Bin')
#             plt.colorbar(im, ax=axes[row, col])

#     stage = '1' if epoch <= 50 else ('2' if epoch <= 150 else '3')
#     plt.suptitle(
#         f'{title} Domain @ Epoch {epoch} | Stage {stage}',
#         fontsize=13
#     )
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(log_dir, f'vis_{title.lower()}_ep{epoch}.png'),
#         dpi=150
#     )
#     plt.close()

# # =====================
# # 5. 动态权重调度
# # =====================
# def get_dynamic_lambda(epoch):
#     """
#     Stage1 (0-49)   : 重构优先，网络先学会基本映射
#     Stage2 (50-149) : 线性过渡，逐步引入 cycle / feat_match
#     Stage3 (150+)   : 正常 CycleGAN，feat_match 主导清晰度
#     """
#     if epoch < 50:
#         return dict(adv=0.5, rec=10.0, cycle=1.0, identity=1,
#                     freq=2.0, smooth=0.1, feat_match=2.0)
#     elif epoch < 150:
#         p = (epoch - 50) / 100.0          # 0→1
#         return dict(
#             adv        = 0.5  + 0.5  * p,
#             rec        = 10.0 - 5.0  * p,
#             cycle      = 1.0  + 7.0  * p,
#             identity   = 1.0  - 0.9  * p,
#             freq       = 2.0  - 1.0  * p,
#             smooth=0.1,
#             feat_match = 2.0  + 2.0  * p,
#         )
#     else:
#         return dict(adv=1.0, rec=5.0, cycle=8.0, identity=0.1,
#                     freq=1.0, smooth=0.1, feat_match=4.0)

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

# lambda_gp              = 10.0
# num_epochs             = 500
# batch_size             = 16
# warmup, N0, N1         = 50, 250, 50
# total_epochs           = 200

# def lr_lambda(epoch):
#     if epoch < warmup:
#         return (epoch + 1) / warmup
#     e = epoch - warmup
#     if e < N0:
#         return 1.0
#     return max(0.0, 1.0 - (e - N0) / float(N1))

# scheduler_G       = lr_scheduler.LambdaLR(optimizer_G,       lr_lambda=lr_lambda)
# scheduler_D_audio = lr_scheduler.LambdaLR(optimizer_D_audio, lr_lambda=lr_lambda)
# scheduler_D_acc   = lr_scheduler.LambdaLR(optimizer_D_acc,   lr_lambda=lr_lambda)

# wgan_loss = WGANLoss()

# # =====================
# # 7. 损失记录容器
# # =====================
# history = dict(
#     d_audio=[], d_acc=[],
#     g_a2acc=[], g_acc2a=[],
#     cycle=[], freq_cycle=[],
#     smooth=[], feat_match=[]
# )
# current_lambdas = {}

# # =====================
# # 8. 训练循环
# # =====================
# print("Starting Training Loop...")
# print(f"Data: ACC & Audio spectrograms, shape (80, 80), normalized to [-1, 1]")
# print(f"Total epochs: {total_epochs}, batch_size: {batch_size}")

# for epoch in range(total_epochs):

#     # ---- 获取本 epoch 动态权重 ----
#     current_lambdas  = get_dynamic_lambda(epoch)
#     lambda_adv       = current_lambdas['adv']
#     lambda_rec       = current_lambdas['rec']
#     lambda_cycle     = current_lambdas['cycle']
#     lambda_identity  = current_lambdas['identity']
#     lambda_freq      = current_lambdas['freq']
#     lambda_smooth    = current_lambdas['smooth']
#     lambda_feat      = current_lambdas['feat_match']

#     if epoch in (0, 50, 150):
#         print(f"\n{'='*60}")
#         print(f"[Epoch {epoch}] Lambda weights →  {current_lambdas}")
#         print(f"{'='*60}\n")

#     # ---- Epoch 累计量 ----
#     ep = {k: 0.0 for k in
#           ['d_audio','d_acc','g_a2acc','g_acc2a',
#            'cycle','freq_cycle','smooth','feat_match']}
#     num_batches       = 0
#     critic_iterations = 5 if epoch < 50 else 2

#     for i, (acc_np, audio_np) in enumerate(
#             data_loader.get_batch_iter(batch_size, dataset='train')):

#         num_batches  += 1
#         current_bs    = acc_np.shape[0]

#         # -------------------------------------------------
#         # A. 数据准备 — 直接转为tensor，数据已经是(B,80,80)归一化到[-1,1]
#         # -------------------------------------------------
#         # 添加channel维度: (B, 80, 80) -> (B, 1, 80, 80)
        
#         audio_mel= torch.from_numpy(audio_np).float().unsqueeze(1).to(device)
#         audio_mel =audio_mel[:,:,:80,:80]
        
#         magnitude= torch.from_numpy(acc_np).float().unsqueeze(1).to(device)
#         magnitude=magnitude[:,:,:80,:80]
        
#         if i == 0 and epoch == 0:
#             print(f"[Shape] audio_mel={audio_mel.shape}  magnitude={magnitude.shape}")

#         # -------------------------------------------------
#         # B. 判别器训练（WGAN-GP）
#         # -------------------------------------------------
#         for _ in range(critic_iterations):

#             # --- D_audio ---
#             optimizer_D_audio.zero_grad()
#             with torch.no_grad():
#                 fake_audio_tmp = G_acc2audio(magnitude)
#             real_v = D_audio(audio_mel)
#             fake_v = D_audio(fake_audio_tmp.detach())
#             d_audio_loss = (wgan_loss.discriminator_loss(real_v, fake_v)
#                             + lambda_gp * compute_gradient_penalty(
#                                 D_audio, audio_mel, fake_audio_tmp, device))
#             d_audio_loss.backward()
#             optimizer_D_audio.step()

#             # --- D_acc ---
#             optimizer_D_acc.zero_grad()
#             with torch.no_grad():
#                 fake_acc_tmp = G_audio2acc(audio_mel)
#             real_v = D_acc(magnitude)
#             fake_v = D_acc(fake_acc_tmp.detach())
#             d_acc_loss = (wgan_loss.discriminator_loss(real_v, fake_v)
#                           + lambda_gp * compute_gradient_penalty(
#                               D_acc, magnitude, fake_acc_tmp, device))
#             d_acc_loss.backward()
#             optimizer_D_acc.step()

#         ep['d_audio'] += d_audio_loss.item()
#         ep['d_acc']   += d_acc_loss.item()

#         # -------------------------------------------------
#         # C. 生成器训练
#         # -------------------------------------------------
#         optimizer_G.zero_grad()

#         # C-1. 前向
#         fake_audio_mel = G_acc2audio(magnitude)   # Acc  → Fake Audio
#         fake_acc_mag   = G_audio2acc(audio_mel)   # Audio → Fake Acc

#         # C-2. 判别器打分（带中间特征）
#         fake_audio_v, fake_audio_feats = D_audio(fake_audio_mel, return_feats=True)
#         real_audio_v, real_audio_feats = D_audio(audio_mel,      return_feats=True)

#         fake_acc_v,   fake_acc_feats   = D_acc(fake_acc_mag, return_feats=True)
#         real_acc_v,   real_acc_feats   = D_acc(magnitude,    return_feats=True)

#         # C-3. 循环一致性
#         recovered_acc_mag   = G_audio2acc(fake_audio_mel)  # Acc→Audio→Acc
#         recovered_audio_mel = G_acc2audio(fake_acc_mag)    # Audio→Acc→Audio

#         # C-4. 身份映射
#         identity_audio = G_acc2audio(audio_mel) if lambda_identity > 0 else None
#         identity_acc   = G_audio2acc(magnitude)  if lambda_identity > 0 else None

#         # C-5. 基础 CycleGAN 损失
#         g_a2acc_total, _, _, g_a2acc_cyc, _ = generator_loss(
#             fake_acc_v, magnitude, fake_acc_mag, recovered_acc_mag,
#             identity_acc, lambda_adv, lambda_rec, lambda_cycle, lambda_identity
#         )
#         g_acc2a_total, _, _, g_acc2a_cyc, _ = generator_loss(
#             fake_audio_v, audio_mel, fake_audio_mel, recovered_audio_mel,
#             identity_audio, lambda_adv, lambda_rec, lambda_cycle, lambda_identity
#         )

#         # C-6. 特征匹配损失
#         feat_audio = feat_match_loss(real_audio_feats, fake_audio_feats) * 2.0
#         feat_acc   = feat_match_loss(real_acc_feats,   fake_acc_feats)
#         feat_total = feat_audio + feat_acc

#         # C-7. 频率感知循环损失（只约束 acc 域）
#         freq_cycle_loss = freq_loss_fn(recovered_acc_mag, magnitude)

#         # C-8. 谱平滑损失（只约束 acc 域，不碰 audio 域）
#         smooth_loss = (spectral_smooth(fake_acc_mag)
#                      + spectral_smooth(recovered_acc_mag))

#         # C-9. 汇总
#         g_total    = (g_a2acc_total
#                     + g_acc2a_total
#                     + lambda_feat   * feat_total
#                     + lambda_freq   * freq_cycle_loss
#                     + lambda_smooth * smooth_loss)
#         cycle_loss = g_a2acc_cyc + g_acc2a_cyc

#         g_total.backward()
#         torch.nn.utils.clip_grad_norm_(
#             list(G_audio2acc.parameters()) + list(G_acc2audio.parameters()),
#             max_norm=1.0
#         )
#         optimizer_G.step()

#         # C-10. 累计
#         ep['g_a2acc']      += g_a2acc_total.item()
#         ep['g_acc2a']      += g_acc2a_total.item()
#         ep['cycle']        += cycle_loss.item()
#         ep['freq_cycle']   += freq_cycle_loss.item()
#         ep['smooth']       += smooth_loss.item()
#         ep['feat_match']   += feat_total.item()

#         # C-11. Batch 日志
#         if i % 5 == 0:
#             stage = ('1-Rec' if epoch < 50
#                      else ('2-Trans' if epoch < 150 else '3-Cycle'))
#             print(
#                 f"[Ep {epoch:>3d}/{total_epochs} | Bt {i:>4d} | {stage}] "
#                 f"D: {d_audio_loss.item():.3f}/{d_acc_loss.item():.3f} | "
#                 f"G: {g_total.item():.3f} "
#                 f"Cyc: {cycle_loss.item():.3f} | "
#                 f"Feat: {feat_total.item():.3f} "
#                 f"(aud={feat_audio.item():.3f} acc={feat_acc.item():.3f}) | "
#                 f"Freq: {freq_cycle_loss.item():.3f} "
#                 f"Smo: {smooth_loss.item():.3f}"
#             )

#     # =====================
#     # 9. Epoch 结束
#     # =====================
#     scheduler_G.step()
#     scheduler_D_audio.step()
#     scheduler_D_acc.step()

#     # 计算平均并记录
#     for k in ep:
#         history[k].append(ep[k] / max(num_batches, 1))

#     lr_now = optimizer_G.param_groups[0]['lr']
#     print(
#         f"\n>>> Ep {epoch+1:>3d}/{total_epochs} | LR={lr_now:.2e} | "
#         f"λ(adv={lambda_adv:.1f} rec={lambda_rec:.1f} "
#         f"cyc={lambda_cycle:.1f} id={lambda_identity:.1f} "
#         f"freq={lambda_freq:.1f} smo={lambda_smooth:.2f} "
#         f"feat={lambda_feat:.1f})\n"
#         f"    D_audio={history['d_audio'][-1]:.4f}  "
#         f"D_acc={history['d_acc'][-1]:.4f}\n"
#         f"    G_a2acc={history['g_a2acc'][-1]:.4f}  "
#         f"G_acc2a={history['g_acc2a'][-1]:.4f}\n"
#         f"    Cycle={history['cycle'][-1]:.4f}  "
#         f"FreqCyc={history['freq_cycle'][-1]:.4f}  "
#         f"Smooth={history['smooth'][-1]:.4f}  "
#         f"FeatMatch={history['feat_match'][-1]:.4f}\n"
#     )

#     # =====================
#     # 10. 定期保存
#     # =====================
#     if (epoch + 1) % 50 == 0:

#         # ---------- 保存权重 ----------
#         save_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
#         torch.save({
#             'epoch'                       : epoch + 1,
#             'G_audio2acc_state_dict'      : G_audio2acc.state_dict(),
#             'G_acc2audio_state_dict'      : G_acc2audio.state_dict(),
#             'D_audio_state_dict'          : D_audio.state_dict(),
#             'D_acc_state_dict'            : D_acc.state_dict(),
#             'optimizer_G_state_dict'      : optimizer_G.state_dict(),
#             'optimizer_D_audio_state_dict': optimizer_D_audio.state_dict(),
#             'optimizer_D_acc_state_dict'  : optimizer_D_acc.state_dict(),
#             'scheduler_G_state_dict'      : scheduler_G.state_dict(),
#             'history'                     : history,
#             'current_lambdas'             : current_lambdas,
#         }, save_path)
#         print(f"[Save] Checkpoint → {save_path}")

#         # ---------- Loss 曲线 ----------
#         fig, axes = plt.subplots(4, 1, figsize=(12, 16))

#         axes[0].set_title('Discriminator Loss')
#         axes[0].plot(history['d_audio'], label='D_Audio', color='steelblue')
#         axes[0].plot(history['d_acc'],   label='D_Acc',   color='darkorange')

#         axes[1].set_title('Generator Loss')
#         axes[1].plot(history['g_a2acc'], label='G_audio2acc', color='green')
#         axes[1].plot(history['g_acc2a'], label='G_acc2audio', color='purple')

#         axes[2].set_title('Cycle Loss')
#         axes[2].plot(history['cycle'],      label='Cycle',    color='red',    lw=2)
#         axes[2].plot(history['freq_cycle'], label='FreqCycle',color='tomato', lw=1.5)
#         axes[2].plot(history['smooth'],     label='Smooth',   color='peru',   lw=1.5)

#         axes[3].set_title('Feature Matching Loss')
#         axes[3].plot(history['feat_match'], label='FeatMatch', color='teal', lw=2)

#         for ax in axes:
#             ax.legend(fontsize=8)
#             ax.grid(True, alpha=0.3)
#             ax.set_xlabel('Epoch')
#             for bd, lbl in zip([50, 150], ['S1→S2', 'S2→S3']):
#                 if bd < len(history['d_audio']):
#                     ax.axvline(x=bd, color='gray', ls='--', alpha=0.6, label=lbl)

#         plt.tight_layout()
#         plt.savefig(os.path.join(log_dir, f'loss_ep{epoch+1}.png'), dpi=150)
#         plt.close()

#         # ---------- 可视化生成效果 ----------
#         G_audio2acc.eval()
#         G_acc2audio.eval()

#         with torch.no_grad():
#             vis_acc_np, vis_audio_np = next(iter(
#                 data_loader.get_batch_iter(4, dataset='val', shuffle=False)
#             ))
#             n_vis = min(4, vis_acc_np.shape[0])
            
#             # 直接转tensor，添加channel维度
#             vis_audio = torch.from_numpy(vis_audio_np[:n_vis]).float().unsqueeze(1).to(device)
#             vis_acc   = torch.from_numpy(vis_acc_np[:n_vis]).float().unsqueeze(1).to(device)

#             v_fake_acc   = G_audio2acc(vis_audio)          # Audio → Acc
#             v_fake_audio = G_acc2audio(vis_acc)            # Acc   → Audio
#             v_rec_acc    = G_audio2acc(v_fake_audio)       # Acc   → Audio → Acc
#             v_rec_audio  = G_acc2audio(v_fake_acc)         # Audio → Acc   → Audio

#         G_audio2acc.train()
#         G_acc2audio.train()

#         def t2np(t):
#             return t.squeeze(1).cpu().numpy()   # [B, F, T]

#         # 可视化 Acc 域
#         _plot_domain(
#             orig  = t2np(vis_acc),
#             pred  = t2np(v_fake_acc),
#             cycl  = t2np(v_rec_acc),
#             title = 'Acc',
#             epoch = epoch + 1,
#             log_dir = log_dir,
#         )

#         # 可视化 Audio（Mel）域
#         _plot_domain(
#             orig  = t2np(vis_audio),
#             pred  = t2np(v_fake_audio),
#             cycl  = t2np(v_rec_audio),
#             title = 'Mel',
#             epoch = epoch + 1,
#             log_dir = log_dir,
#         )

#         print(f"[Plot] Saved visualizations for epoch {epoch+1}\n")


# # =====================
# # 12. 保存最终模型
# # =====================
# final_path = os.path.join(model_dir, 'final_models.pth')
# torch.save({
#     'G_audio2acc_state_dict': G_audio2acc.state_dict(),
#     'G_acc2audio_state_dict': G_acc2audio.state_dict(),
#     'D_audio_state_dict'    : D_audio.state_dict(),
#     'D_acc_state_dict'      : D_acc.state_dict(),
#     'history'               : history,
# }, final_path)
# print(f"\nTraining completed.  Final model → {final_path}")
