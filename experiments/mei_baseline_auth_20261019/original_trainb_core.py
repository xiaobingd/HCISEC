"""Unmodified class definitions extracted from user modelb.py.
Source SHA256: 05c7c8e40e427e5529346bd360940ef35a7eebd028c421b3fc119d37e1f6f153
Classes retained verbatim; only module imports narrowed to torch.
"""
import torch
from torch import nn
from torch.nn import functional as F

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
