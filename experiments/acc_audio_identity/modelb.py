"""
模型定义 + 损失函数

多源特征融合的 1:N 身份认证模型 & 对比学习损失函数

三个特征源（全部基于 CycleGAN）：

  特征源1 — 原始 ACC 浅层特征:
    CycleGAN 编码器浅层 model[:7] → (128, 40, 40) → GAP → (128,)
    捕获原始 ACC 频谱图的底层纹理/频率结构

  特征源2 — CycleGAN 瓶颈特征:
    CycleGAN 编码器深层 model[:14] (AxialAIA 后) → (256, 20, 20) → GAP+GStd → (512,)
    捕获 CycleGAN 学到的高层语义/跨模态映射特征

  特征源3 — Cycle 重建差异特征:
    ACC → G_acc2audio → fake_audio → G_audio2acc → recovered_ACC
    diff = ACC - recovered_ACC → 通过编码器浅层 → GAP → (128,)
    捕获用户特异性的重建误差模式（不同用户骨传导特性不同，重建误差模式不同）

融合:
  concat(shallow_feat, bottleneck_feat, diff_feat) → MLP → (512,)
  → 投影头 → (128,) → SupCon Loss [训练]
  → L2 归一化 → 余弦相似度 [推理]

损失函数:
  1. SupConLoss — 监督对比损失（有用户标签时）
  2. SimCLRLoss — 自监督对比损失（无标签时，用增强正对）
  3. ArcFaceLoss — 角度间隔分类损失
  4. CombinedAuthLoss — 组合损失（对比 + 分类）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加 cyclegan_bo 到路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CYCLEGAN_DIR = os.path.join(_SCRIPT_DIR, '..', 'cyclegan_bo')
if _CYCLEGAN_DIR not in sys.path:
    sys.path.insert(0, _CYCLEGAN_DIR)

from model_drop import GeneratorAccToAudio, GeneratorAudioToAcc


# ============================================================
# 1. 特征聚合器
# ============================================================

class FeatureAggregator(nn.Module):
    """将 (B, C, H, W) 特征图聚合为 (B, D) 向量"""
    def __init__(self, in_channels, strategy='gap_std'):
        super().__init__()
        self.strategy = strategy
        if strategy == 'gap':
            self.out_dim = in_channels
        elif strategy in ('gap_std', 'gap_max'):
            self.out_dim = in_channels * 2
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, x):
        B, C = x.shape[:2]
        gap = x.view(B, C, -1).mean(dim=2)
        if self.strategy == 'gap':
            return gap
        elif self.strategy == 'gap_std':
            gstd = x.view(B, C, -1).std(dim=2)
            return torch.cat([gap, gstd], dim=1)
        else:  # gap_max
            gmax = x.view(B, C, -1).max(dim=2).values
            return torch.cat([gap, gmax], dim=1)


# ============================================================
# 2. 投影头
# ============================================================

class ProjectionHead(nn.Module):
    """MLP 投影头：训练时用，推理时不用"""
    def __init__(self, feat_dim, hidden_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


# ============================================================
# 3. 多源特征融合认证模型
# ============================================================

class MultiSourceAuthModel(nn.Module):
    """
    多源特征融合身份认证模型

    三个特征源全部来自 CycleGAN 生成器（冻结），
    只训练融合层和投影头。

    Args:
        g_acc2audio:    CycleGAN ACC→Audio 生成器
        g_audio2acc:    CycleGAN Audio→ACC 生成器
        freeze_generators: 是否冻结生成器参数
        use_shallow:    是否使用浅层特征（特征源1）
        use_bottleneck: 是否使用瓶颈特征（特征源2）
        use_cycle_diff: 是否使用重建差异特征（特征源3）
        bottleneck_cut: 瓶颈特征截取位置（14=AxialAIA后）
        shallow_cut:    浅层特征截取位置（7=下采样1次后）
        proj_dim:       投影维度
    """

    def __init__(self, g_acc2audio, g_audio2acc,
                 freeze_generators=True,
                 use_shallow=True,
                 use_bottleneck=True,
                 use_cycle_diff=True,
                 bottleneck_cut=14,
                 shallow_cut=7,
                 proj_dim=128):
        super().__init__()

        self.use_shallow    = use_shallow
        self.use_bottleneck = use_bottleneck
        self.use_cycle_diff = use_cycle_diff

        # ---- 保存完整生成器（用于 cycle 重建差异） ----
        self.g_acc2audio = g_acc2audio
        self.g_audio2acc = g_audio2acc

        if freeze_generators:
            for p in self.g_acc2audio.parameters():
                p.requires_grad = False
            for p in self.g_audio2acc.parameters():
                p.requires_grad = False
            self.g_acc2audio.eval()
            self.g_audio2acc.eval()

        # ---- U-Net encoder features from the trained ACC->Audio generator ----
        # shallow: init_conv + down1 -> 128x40x40
        # bottleneck: init_conv + down1 + down2 + residual/attention stack
        self._unet_encoder = hasattr(g_acc2audio, 'init_conv')
        if not self._unet_encoder:
            raise TypeError('Expected the aligned U-Net ACC->Audio generator')

        if use_shallow:
            self.shallow_encoder = nn.Sequential(
                g_acc2audio.init_conv, g_acc2audio.down1)
            self.shallow_agg = FeatureAggregator(
                in_channels=128, strategy='gap')

        if use_bottleneck:
            self.bottleneck_encoder = nn.Sequential(
                g_acc2audio.init_conv, g_acc2audio.down1,
                g_acc2audio.down2, g_acc2audio.bottleneck)
            self.bottleneck_agg = FeatureAggregator(
                in_channels=256, strategy='gap_std')

        if use_cycle_diff:
            self.diff_encoder = nn.Sequential(
                g_acc2audio.init_conv, g_acc2audio.down1)
            self.diff_agg = FeatureAggregator(
                in_channels=128, strategy='gap')
            self.diff_stat_dim = 4

        # ---- 计算融合维度 ----
        fusion_dim = 0
        if use_shallow:
            fusion_dim += self.shallow_agg.out_dim
        if use_bottleneck:
            fusion_dim += self.bottleneck_agg.out_dim
        if use_cycle_diff:
            fusion_dim += self.diff_agg.out_dim + self.diff_stat_dim

        self.feat_dim = 512

        # ---- 融合层（可训练） ----
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
        )

        # ---- 投影头（可训练） ----
        self.proj_dim = proj_dim
        self.projector = ProjectionHead(self.feat_dim, self.feat_dim // 2, proj_dim)

        self._config = {
            'use_shallow': use_shallow,
            'use_bottleneck': use_bottleneck,
            'use_cycle_diff': use_cycle_diff,
            'bottleneck_cut': bottleneck_cut,
            'shallow_cut': shallow_cut,
            'fusion_dim': fusion_dim,
            'feat_dim': self.feat_dim,
            'proj_dim': proj_dim,
        }

    def _get_channels(self, cut_at):
        if cut_at <= 4:
            return 64
        elif cut_at <= 7:
            return 128
        else:
            return 256

    @torch.no_grad()
    def _compute_cycle_diff(self, acc):
        fake_audio    = self.g_acc2audio(acc)
        recovered_acc = self.g_audio2acc(fake_audio)
        diff = acc - recovered_acc

        B = diff.shape[0]
        diff_flat = diff.view(B, -1)
        stats = torch.stack([
            diff_flat.mean(dim=1),
            diff_flat.std(dim=1),
            diff_flat.abs().max(dim=1).values,
            (diff_flat ** 2).mean(dim=1).sqrt(),
        ], dim=1)

        return diff, stats

    def forward(self, acc, return_embedding=False):
        features = []

        if self.use_shallow:
            with torch.no_grad():
                shallow_feat_map = self.shallow_encoder(acc)
            shallow_feat = self.shallow_agg(shallow_feat_map)
            features.append(shallow_feat)

        if self.use_bottleneck:
            with torch.no_grad():
                bottleneck_feat_map = self.bottleneck_encoder(acc)
            bottleneck_feat = self.bottleneck_agg(bottleneck_feat_map)
            features.append(bottleneck_feat)

        if self.use_cycle_diff:
            diff_map, diff_stats = self._compute_cycle_diff(acc)
            with torch.no_grad():
                diff_feat_map = self.diff_encoder(diff_map)
            diff_feat = self.diff_agg(diff_feat_map)
            features.append(diff_feat)
            features.append(diff_stats)

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        if return_embedding:
            return F.normalize(fused, p=2, dim=1)

        return self.projector(fused)

    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_embedding=True)

    def train(self, mode=True):
        super().train(mode)
        self.g_acc2audio.eval()
        self.g_audio2acc.eval()
        return self

    def get_config(self):
        return self._config.copy()


# ============================================================
# 4. 单源模型（向后兼容，简化版）
# ============================================================

class SingleSourceAuthModel(nn.Module):
    def __init__(self, generator, cut_at=14, freeze=True,
                 agg_strategy='gap_std', proj_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(*list(generator.model.children())[:cut_at])
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.agg = FeatureAggregator(256, agg_strategy)
        self.feat_dim = self.agg.out_dim
        self.proj_dim = proj_dim
        self.projector = ProjectionHead(self.feat_dim, self.feat_dim // 2, proj_dim)

    def forward(self, x, return_embedding=False):
        feat_map = self.encoder(x)
        feat_vec = self.agg(feat_map)
        if return_embedding:
            return F.normalize(feat_vec, p=2, dim=1)
        return self.projector(feat_vec)

    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_embedding=True)

    def train(self, mode=True):
        if all(not p.requires_grad for p in self.encoder.parameters()):
            super().train(mode)
            self.encoder.eval()
            return self
        return super().train(mode)


# ============================================================
# 5. 工厂函数
# ============================================================

def create_multi_source_model(checkpoint_path,
                               use_shallow=True,
                               use_bottleneck=True,
                               use_cycle_diff=True,
                               freeze_generators=True,
                               proj_dim=128,
                               device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    g_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1)
    g_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1)

    def _find_key(ckpt, candidates):
        for k in candidates:
            if k in ckpt:
                return k
        raise KeyError(f"找不到生成器key，checkpoint包含: {list(ckpt.keys())}")

    key_a2b = _find_key(ckpt, [
        'G_acc2audio_state_dict', 'g_acc2audio_state_dict',
        'G_A2B_state_dict', 'g_A2B_state_dict',
        'generator_A2B', 'G_A2B',
    ])
    key_b2a = _find_key(ckpt, [
        'G_audio2acc_state_dict', 'g_audio2acc_state_dict',
        'G_B2A_state_dict', 'g_B2A_state_dict',
        'generator_B2A', 'G_B2A',
    ])
    g_acc2audio.load_state_dict(ckpt[key_a2b])
    g_audio2acc.load_state_dict(ckpt[key_b2a])

    model = MultiSourceAuthModel(
        g_acc2audio=g_acc2audio,
        g_audio2acc=g_audio2acc,
        freeze_generators=freeze_generators,
        use_shallow=use_shallow,
        use_bottleneck=use_bottleneck,
        use_cycle_diff=use_cycle_diff,
        proj_dim=proj_dim,
    ).to(device)

    info = {
        'epoch': ckpt.get('epoch', '?'),
        'sources': [],
        'feat_dim': model.feat_dim,
        'proj_dim': proj_dim,
    }
    if use_shallow:    info['sources'].append('shallow(128d)')
    if use_bottleneck: info['sources'].append('bottleneck(512d)')
    if use_cycle_diff: info['sources'].append('cycle_diff(128d+4d)')

    return model, info


def create_single_source_model(checkpoint_path,
                                generator_key='G_acc2audio_state_dict',
                                freeze=True,
                                agg_strategy='gap_std',
                                proj_dim=128,
                                device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'acc2audio' in generator_key:
        gen = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1)
    else:
        gen = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1)
    gen.load_state_dict(ckpt[generator_key])

    model = SingleSourceAuthModel(gen, cut_at=14, freeze=freeze,
                                   agg_strategy=agg_strategy,
                                   proj_dim=proj_dim).to(device)

    return model, {'epoch': ckpt.get('epoch', '?'),
                   'feat_dim': model.feat_dim, 'proj_dim': proj_dim}


# ============================================================
# 6. 损失函数
# ============================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)
    
    参考: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    
    同一用户的所有样本互为正对，不同用户的样本为负对。
    
    Args:
        temperature: 温度参数，控制分布锐度（越小越锐利）
        base_temperature: 基准温度，用于缩放
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        similarity = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (num_positives + 1e-12)

        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob[valid]
        return loss.mean()


class SimCLRLoss(nn.Module):
    """
    SimCLR 自监督对比损失（NT-Xent）
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        device = z_i.device

        z = torch.cat([z_i, z_j], dim=0)
        N = 2 * batch_size

        sim = torch.matmul(z, z.T) / self.temperature

        mask_pos = torch.zeros(N, N, device=device)
        for i in range(batch_size):
            mask_pos[i, i + batch_size] = 1
            mask_pos[i + batch_size, i] = 1

        mask_self = torch.eye(N, device=device).bool()
        sim.masked_fill_(mask_self, -1e9)

        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1) - exp_sim.diag()
        
        pos_sim = (mask_pos * sim).sum(dim=1)

        loss = -pos_sim + torch.log(denom + 1e-12)
        return loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace 损失 — 角度间隔分类损失
    """
    def __init__(self, embed_dim=512, num_classes=10, margin=0.5, scale=30.0):
        super().__init__()
        self.margin      = margin
        self.scale       = scale
        self.num_classes = num_classes
        self.weight      = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.tensor(margin).cos()
        self.sin_m = torch.tensor(margin).sin()
        self.th    = torch.tensor(margin).cos() * (-1)
        self.mm    = torch.tensor(margin).sin() * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight     = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)
        sine   = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th.to(cosine.device), phi, 
                          cosine - self.mm.to(cosine.device))

        one_hot = F.one_hot(labels, self.num_classes).float()
        output  = one_hot * phi + (1.0 - one_hot) * cosine
        output  = output * self.scale

        return F.cross_entropy(output, labels)


class CombinedAuthLoss(nn.Module):
    """
    组合损失：SupCon + ArcFace（可选）
    """
    def __init__(self, embed_dim=512, num_classes=10, 
                 supcon_weight=1.0, arcface_weight=0.5,
                 temperature=0.07):
        super().__init__()
        self.supcon_loss  = SupConLoss(temperature=temperature)
        self.arcface_loss = ArcFaceLoss(embed_dim, num_classes) if num_classes > 1 else None
        self.supcon_weight  = supcon_weight
        self.arcface_weight = arcface_weight

    def forward(self, projections, embeddings, labels):
        loss_supcon = self.supcon_loss(projections, labels)
        total = self.supcon_weight * loss_supcon

        loss_dict = {'supcon': loss_supcon.item()}

        if self.arcface_loss is not None and self.arcface_weight > 0:
            loss_arc = self.arcface_loss(embeddings, labels)
            total = total + self.arcface_weight * loss_arc
            loss_dict['arcface'] = loss_arc.item()

        loss_dict['total'] = total.item()
        return total, loss_dict



# ============================================================
# 7. Diffusion 特征提取器
# ============================================================

class DiffusionFeatureExtractor(nn.Module):
    """
    从Diffusion模型中提取身份特征

    支持的特征源（基于discover_diff_features.py的结果）:
      F1: A2U UNet瓶颈特征 (256, 20, 20) → GAP → (256,)
      F4: 链式重建差异 |ACC - rec_ACC| (1, 80, 80) → 统计量 → (stat_dim,)
      F5: 噪声预测模式 (1, 80, 80) → 浅层编码 → GAP → (feat_dim,)

    Args:
        model_a2u: Diffusion ACC→Audio 模型
        model_u2a: Diffusion Audio→ACC 模型
        schedule:  DiffusionSchedule 实例
        use_f1:    是否使用A2U瓶颈特征
        use_f4:    是否使用链式重建差异
        use_f5:    是否使用噪声预测模式
        ddim_steps: DDIM采样步数
        noise_t:   F5噪声预测的时间步
    """

    def __init__(self, model_a2u, model_u2a, schedule,
                 use_f1=True, use_f4=True, use_f5=True,
                 ddim_steps=50, noise_t=500,
                 freeze=True):
        super().__init__()
        self.model_a2u = model_a2u
        self.model_u2a = model_u2a
        self.schedule = schedule
        self.use_f1 = use_f1
        self.use_f4 = use_f4
        self.use_f5 = use_f5
        self.ddim_steps = ddim_steps
        self.noise_t = noise_t

        if freeze:
            for p in self.model_a2u.parameters():
                p.requires_grad = False
            for p in self.model_u2a.parameters():
                p.requires_grad = False
            self.model_a2u.eval()
            self.model_u2a.eval()

        # F1: 瓶颈特征聚合 (256, 20, 20) → (256,)
        self.out_dim = 0
        if use_f1:
            self.f1_agg = FeatureAggregator(256, strategy='gap')
            self.out_dim += self.f1_agg.out_dim  # 256

        # F4: 重建差异统计量
        if use_f4:
            self.f4_stat_dim = 8  # mean, std, max, rms, p25, p50, p75, p95
            self.out_dim += self.f4_stat_dim

        # F5: 噪声预测模式编码
        if use_f5:
            self.f5_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.f5_dim = 32
            self.out_dim += self.f5_dim

    @torch.no_grad()
    def _extract_f1(self, acc):
        """F1: A2U UNet瓶颈特征"""
        B = acc.size(0)
        t_zero = torch.zeros(B, device=acc.device, dtype=torch.long)
        dummy = torch.zeros_like(acc)
        _, feats = self.model_a2u(dummy, t_zero, acc, return_features=True)
        return self.f1_agg(feats['bottleneck'])  # (B, 256)

    @torch.no_grad()
    def _extract_f4(self, acc):
        """F4: 链式重建差异统计量"""
        B = acc.size(0)
        shape = (B, 1, acc.size(2), acc.size(3))
        fake_audio = self.schedule.ddim_sample(
            self.model_a2u, acc, shape, self.ddim_steps)
        rec_acc = self.schedule.ddim_sample(
            self.model_u2a, fake_audio, shape, self.ddim_steps)
        diff = torch.abs(acc - rec_acc)
        diff_flat = diff.reshape(B, -1)
        stats = torch.stack([
            diff_flat.mean(dim=1),
            diff_flat.std(dim=1),
            diff_flat.max(dim=1).values,
            (diff_flat ** 2).mean(dim=1).sqrt(),
            torch.quantile(diff_flat, 0.25, dim=1),
            torch.quantile(diff_flat, 0.50, dim=1),
            torch.quantile(diff_flat, 0.75, dim=1),
            torch.quantile(diff_flat, 0.95, dim=1),
        ], dim=1)  # (B, 8)
        return stats

    @torch.no_grad()
    def _extract_f5_raw(self, acc):
        """F5: 噪声预测模式（原始）"""
        B = acc.size(0)
        t = torch.full((B,), self.noise_t, device=acc.device, dtype=torch.long)
        x_t, _ = self.schedule.forward_diffusion(acc, t)
        eps_pred = self.model_a2u(x_t, t, acc)
        return eps_pred  # (B, 1, 80, 80)

    def forward(self, acc):
        """提取Diffusion特征并拼接"""
        features = []

        if self.use_f1:
            features.append(self._extract_f1(acc))

        if self.use_f4:
            features.append(self._extract_f4(acc))

        if self.use_f5:
            raw_f5 = self._extract_f5_raw(acc)
            f5_feat = self.f5_encoder(raw_f5).squeeze(-1).squeeze(-1)
            features.append(f5_feat)

        return torch.cat(features, dim=1)  # (B, out_dim)

    def train(self, mode=True):
        super().train(mode)
        self.model_a2u.eval()
        self.model_u2a.eval()
        return self


# ============================================================
# 8. V2认证模型: CycleGAN + Diffusion 融合
# ============================================================

class MultiSourceAuthModelV2(nn.Module):
    """
    V2: 在原有CycleGAN特征基础上，融合Diffusion特征

    CycleGAN特征 (原有):
      - shallow:    (128,) — 浅层纹理
      - bottleneck: (512,) — 高层语义
      - cycle_diff: (132,) — 重建差异

    Diffusion特征 (新增):
      - F1: A2U瓶颈 (256,)
      - F4: 重建差异统计 (8,)
      - F5: 噪声模式 (32,)

    总融合: concat → MLP → 512维嵌入
    """

    def __init__(self, g_acc2audio, g_audio2acc,
                 diff_extractor,
                 freeze_generators=True,
                 use_shallow=True,
                 use_bottleneck=True,
                 use_cycle_diff=True,
                 bottleneck_cut=14,
                 shallow_cut=7,
                 proj_dim=128):
        super().__init__()

        self.use_shallow = use_shallow
        self.use_bottleneck = use_bottleneck
        self.use_cycle_diff = use_cycle_diff

        # ---- CycleGAN生成器 ----
        self.g_acc2audio = g_acc2audio
        self.g_audio2acc = g_audio2acc
        if freeze_generators:
            for p in self.g_acc2audio.parameters():
                p.requires_grad = False
            for p in self.g_audio2acc.parameters():
                p.requires_grad = False
            self.g_acc2audio.eval()
            self.g_audio2acc.eval()

        # ---- CycleGAN特征提取 (同V1) ----
        encoder_modules = list(g_acc2audio.model.children())

        if use_shallow:
            self.shallow_encoder = nn.Sequential(*encoder_modules[:shallow_cut])
            ch = 128 if shallow_cut <= 7 else 256
            self.shallow_agg = FeatureAggregator(ch, strategy='gap')

        if use_bottleneck:
            self.bottleneck_encoder = nn.Sequential(*encoder_modules[:bottleneck_cut])
            self.bottleneck_agg = FeatureAggregator(256, strategy='gap_std')

        if use_cycle_diff:
            self.diff_encoder = nn.Sequential(*encoder_modules[:shallow_cut])
            ch = 128 if shallow_cut <= 7 else 256
            self.diff_agg = FeatureAggregator(ch, strategy='gap')
            self.diff_stat_dim = 4

        # ---- Diffusion特征提取 ----
        self.diff_extractor = diff_extractor

        # ---- 计算融合维度 ----
        fusion_dim = 0
        if use_shallow:
            fusion_dim += self.shallow_agg.out_dim
        if use_bottleneck:
            fusion_dim += self.bottleneck_agg.out_dim
        if use_cycle_diff:
            fusion_dim += self.diff_agg.out_dim + self.diff_stat_dim
        fusion_dim += self.diff_extractor.out_dim  # Diffusion特征

        self.feat_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
        )

        self.proj_dim = proj_dim
        self.projector = ProjectionHead(self.feat_dim, self.feat_dim // 2, proj_dim)

        self._config = {
            'use_shallow': use_shallow,
            'use_bottleneck': use_bottleneck,
            'use_cycle_diff': use_cycle_diff,
            'diff_features': {
                'use_f1': diff_extractor.use_f1,
                'use_f4': diff_extractor.use_f4,
                'use_f5': diff_extractor.use_f5,
                'diff_out_dim': diff_extractor.out_dim,
            },
            'fusion_dim': fusion_dim,
            'feat_dim': self.feat_dim,
            'proj_dim': proj_dim,
        }

    @torch.no_grad()
    def _compute_cycle_diff(self, acc):
        fake_audio = self.g_acc2audio(acc)
        recovered_acc = self.g_audio2acc(fake_audio)
        diff = acc - recovered_acc
        B = diff.shape[0]
        diff_flat = diff.reshape(B, -1)
        stats = torch.stack([
            diff_flat.mean(dim=1),
            diff_flat.std(dim=1),
            diff_flat.abs().max(dim=1).values,
            (diff_flat ** 2).mean(dim=1).sqrt(),
        ], dim=1)
        return diff, stats

    def forward(self, acc, return_embedding=False):
        features = []

        # CycleGAN特征
        if self.use_shallow:
            with torch.no_grad():
                shallow_feat_map = self.shallow_encoder(acc)
            features.append(self.shallow_agg(shallow_feat_map))

        if self.use_bottleneck:
            with torch.no_grad():
                bottleneck_feat_map = self.bottleneck_encoder(acc)
            features.append(self.bottleneck_agg(bottleneck_feat_map))

        if self.use_cycle_diff:
            diff_map, diff_stats = self._compute_cycle_diff(acc)
            with torch.no_grad():
                diff_feat_map = self.diff_encoder(diff_map)
            features.append(self.diff_agg(diff_feat_map))
            features.append(diff_stats)

        # Diffusion特征
        diff_feat = self.diff_extractor(acc)
        features.append(diff_feat)

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        if return_embedding:
            return F.normalize(fused, p=2, dim=1)
        return self.projector(fused)

    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_embedding=True)

    def train(self, mode=True):
        super().train(mode)
        self.g_acc2audio.eval()
        self.g_audio2acc.eval()
        self.diff_extractor.train(mode)
        return self

    def get_config(self):
        return self._config.copy()


# ============================================================
# 9. V2工厂函数
# ============================================================

def create_multi_source_model_v2(cyclegan_ckpt, diff_ckpt,
                                  use_shallow=True,
                                  use_bottleneck=True,
                                  use_cycle_diff=True,
                                  use_f1=True, use_f4=True, use_f5=True,
                                  ddim_steps=50, timesteps=1000,
                                  proj_dim=128, device='cpu'):
    """
    创建V2融合模型

    Args:
        cyclegan_ckpt: CycleGAN checkpoint路径
        diff_ckpt:     Diffusion checkpoint路径
        use_f1/f4/f5:  选择哪些Diffusion特征
    """
    # ---- 加载CycleGAN ----
    ckpt_cg = torch.load(cyclegan_ckpt, map_location=device, weights_only=False)
    g_acc2audio = GeneratorAccToAudio(input_dim=1, hidden_dim=64, output_dim=1)
    g_audio2acc = GeneratorAudioToAcc(input_dim=1, hidden_dim=64, output_dim=1)

    def _find_key(ckpt, candidates):
        for k in candidates:
            if k in ckpt:
                return k
        raise KeyError(f"找不到key，checkpoint包含: {list(ckpt.keys())}")

    key_a2b = _find_key(ckpt_cg, [
        'G_acc2audio_state_dict', 'g_acc2audio_state_dict',
        'G_A2B_state_dict', 'g_A2B_state_dict',
    ])
    key_b2a = _find_key(ckpt_cg, [
        'G_audio2acc_state_dict', 'g_audio2acc_state_dict',
        'G_B2A_state_dict', 'g_B2A_state_dict',
    ])
    g_acc2audio.load_state_dict(ckpt_cg[key_a2b])
    g_audio2acc.load_state_dict(ckpt_cg[key_b2a])

    # ---- 加载Diffusion ----
    from model_diff import BiometricUNet64
    ckpt_df = torch.load(diff_ckpt, map_location=device, weights_only=False)
    model_a2u = BiometricUNet64(input_dim=1, condition_dim=1).to(device)
    model_u2a = BiometricUNet64(input_dim=1, condition_dim=1).to(device)
    model_a2u.load_state_dict(ckpt_df['model_a2u_state_dict'])
    model_u2a.load_state_dict(ckpt_df['model_u2a_state_dict'])

    from train_ddpm import DiffusionSchedule
    schedule = DiffusionSchedule(timesteps, device=device)

    # ---- 创建Diffusion特征提取器 ----
    diff_extractor = DiffusionFeatureExtractor(
        model_a2u, model_u2a, schedule,
        use_f1=use_f1, use_f4=use_f4, use_f5=use_f5,
        ddim_steps=ddim_steps, freeze=True,
    )

    # ---- 创建V2模型 ----
    model = MultiSourceAuthModelV2(
        g_acc2audio=g_acc2audio,
        g_audio2acc=g_audio2acc,
        diff_extractor=diff_extractor,
        freeze_generators=True,
        use_shallow=use_shallow,
        use_bottleneck=use_bottleneck,
        use_cycle_diff=use_cycle_diff,
        proj_dim=proj_dim,
    ).to(device)

    info = {
        'cyclegan_epoch': ckpt_cg.get('epoch', '?'),
        'diff_epoch': ckpt_df.get('epoch', '?'),
        'sources': [],
        'fusion_dim': model._config['fusion_dim'],
        'feat_dim': model.feat_dim,
    }
    if use_shallow:    info['sources'].append('CG_shallow')
    if use_bottleneck: info['sources'].append('CG_bottleneck')
    if use_cycle_diff: info['sources'].append('CG_cycle_diff')
    if use_f1: info['sources'].append('Diff_F1_bottleneck')
    if use_f4: info['sources'].append('Diff_F4_cycle_diff')
    if use_f5: info['sources'].append('Diff_F5_noise')

    print(f"MultiSourceAuthModelV2 创建完成:")
    print(f"  特征源: {info['sources']}")
    print(f"  融合维度: {info['fusion_dim']} → {model.feat_dim}")
    print(f"  Diffusion特征维度: {diff_extractor.out_dim}")

    return model, info

