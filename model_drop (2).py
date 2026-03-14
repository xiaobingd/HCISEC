import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN


# ==========================================
#  核心组件: 轴向注意力 (Axial Attention)
# ==========================================
class AxialAIA(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (channels // num_heads) ** -0.5

        self.qkv_time  = nn.Linear(channels, channels * 3)
        self.qkv_freq  = nn.Linear(channels, channels * 3)
        self.proj_time = nn.Linear(channels, channels)
        self.proj_freq = nn.Linear(channels, channels)
        self.norm      = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Time Axis Attention (W轴)
        xt  = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        qkv = self.qkv_time(xt).chunk(3, dim=-1)
        q, k, v = [t.view(B * H, W, self.num_heads, C // self.num_heads)
                    .transpose(1, 2) for t in qkv]
        attn     = (q @ k.transpose(-2, -1)) * self.scale
        out_time = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B * H, W, C)
        out_time = self.proj_time(out_time).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Freq Axis Attention (H轴)
        xf  = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        qkv = self.qkv_freq(xf).chunk(3, dim=-1)
        q, k, v = [t.view(B * W, H, self.num_heads, C // self.num_heads)
                    .transpose(1, 2) for t in qkv]
        attn     = (q @ k.transpose(-2, -1)) * self.scale
        out_freq = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B * W, H, C)
        out_freq = self.proj_freq(out_freq).reshape(B, W, H, C).permute(0, 3, 2, 1)

        out = x + 0.5 * (out_time + out_freq)
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ==========================================
#  ResNet 残差块（支持 Dropout）
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        # Dropout 加在两个卷积之间
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


# ==========================================
#  ResNet 生成器（适配 64x64，支持 Dropout）
# ==========================================
class ResNetGenerator(nn.Module):
    """
    针对 64x64 输入设计的 ResNet 生成器。
    结构: 初始卷积 -> 2次下采样 -> N个残差块(中间插入 AxialAIA) -> 2次上采样 -> 输出
    dropout: 残差块内的 Dropout 概率，防止过拟合
    """
    def __init__(self, input_dim=1, output_dim=1,
                 hidden_dim=64, num_residual_blocks=6, dropout=0.0):
        super().__init__()

        # --- 1. 初始卷积 64x64 -> 64x64 ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, padding=0),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        # --- 2. 下采样 ---
        # 64x64 -> 32x32
        model += [
            nn.Conv2d(hidden_dim, hidden_dim * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        ]
        # 32x32 -> 16x16
        model += [
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        ]

        # --- 3. 残差块 + 中间插入 AxialAIA ---
        # 在 16x16 特征图上操作
        for i in range(num_residual_blocks):
            model += [ResidualBlock(hidden_dim * 4, dropout=dropout)]
            # 在中间位置插入轴向注意力
            if i == num_residual_blocks // 2 - 1:
                model += [AxialAIA(channels=hidden_dim * 4, num_heads=4)]

        # --- 4. 上采样 ---
        # 16x16 -> 32x32
        model += [
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        ]
        # 32x32 -> 64x64
        model += [
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        # --- 5. 输出层 ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=7, padding=0),
            nn.Tanh(),   # 输出范围 [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ==========================================
#  兼容训练脚本的包装类
# ==========================================
class GeneratorAccToAudio(ResNetGenerator):
    """Acc -> Audio 生成器"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1,
                 dropout=0.0, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=6,
            dropout=dropout,
        )


class GeneratorAudioToAcc(ResNetGenerator):
    """Audio -> Acc 生成器"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1,
                 dropout=0.0, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=6,
            dropout=dropout,
        )


# ==========================================
#  判别器（PatchGAN，支持中间特征提取）
# ==========================================
def d_block(ic, oc, k=4, s=2, p=1, use_sn=False):
    conv = nn.Conv2d(ic, oc, k, s, p)
    if use_sn:
        conv = SN(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=False))


class SingleScaleDiscriminator(nn.Module):
    """
    输入 64x64 的单尺度 PatchGAN 判别器。
    支持 return_feats 返回中间层特征，用于特征匹配损失。
    """
    def __init__(self, in_ch=1, base=64, use_sn=False):
        super().__init__()
        C = base
        # 每层独立定义，方便提取中间特征
        self.layer1 = d_block(in_ch, C,   k=4, s=2, p=1, use_sn=use_sn)  # 64->32
        self.layer2 = d_block(C,   C*2,   k=4, s=2, p=1, use_sn=use_sn)  # 32->16
        self.layer3 = d_block(C*2, C*4,   k=4, s=2, p=1, use_sn=use_sn)  # 16->8
        self.layer4 = d_block(C*4, C*8,   k=4, s=1, p=1, use_sn=use_sn)  #  8->7
        out_conv    = nn.Conv2d(C*8, 1, kernel_size=4, stride=1, padding=1)
        self.layer5 = SN(out_conv) if use_sn else out_conv                 #  7->6

    def forward(self, x, return_feats=False):
        f1  = self.layer1(x)
        f2  = self.layer2(f1)
        f3  = self.layer3(f2)
        f4  = self.layer4(f3)
        out = self.layer5(f4)
        if return_feats:
            return out, [f1, f2, f3, f4]
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器，对原始及下采样图像分别判别后取均值。
    n_scales=1 时退化为单尺度。
    """
    def __init__(self, in_ch=1, base=64, n_scales=1, use_sn=False):
        super().__init__()
        self.n_scales = n_scales
        self.pool     = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )
        self.discriminators = nn.ModuleList([
            SingleScaleDiscriminator(in_ch=in_ch, base=base, use_sn=use_sn)
            for _ in range(n_scales)
        ])

    def forward(self, x, return_feats=False):
        # 构建多尺度输入列表
        xs = [x]
        for _ in range(1, self.n_scales):
            xs.append(self.pool(xs[-1]))

        outs      = []
        all_feats = []

        for xi, Di in zip(xs, self.discriminators):
            if return_feats:
                out, feats = Di(xi, return_feats=True)
                outs.append(out)
                all_feats.extend(feats)
            else:
                outs.append(Di(xi))

        # 多尺度打分：对齐到第一个尺度后取均值
        target_size = outs[0].shape[-2:]
        outs_up = [outs[0]] + [
            F.interpolate(o, size=target_size,
                          mode='bilinear', align_corners=False)
            for o in outs[1:]
        ]
        merged = torch.stack(outs_up, dim=0).mean(dim=0)

        if return_feats:
            return merged, all_feats
        return merged


# ---- 兼容旧接口 ----
class DiscriminatorAudio(MultiScaleDiscriminator):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1,
                 mel_dim=64, base=64, n_scales=1,
                 use_spectral_norm=False, **kwargs):
        super().__init__(
            in_ch=input_dim, base=base,
            n_scales=n_scales, use_sn=use_spectral_norm
        )


class DiscriminatorAcc(MultiScaleDiscriminator):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1,
                 base=64, n_scales=1,
                 use_spectral_norm=False, **kwargs):
        super().__init__(
            in_ch=input_dim, base=base,
            n_scales=n_scales, use_sn=use_spectral_norm
        )
