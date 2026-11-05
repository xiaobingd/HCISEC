"""
多用户频谱图数据集 + 数据增强

合并自 dataset.py 和 augment.py

包含:
  数据集:
    1. MultiUserSpectrogramDataset — 多用户频谱图数据集
    2. ContrastivePairDataset — 对比学习配对数据集
    3. DualModalDataset — 双模态数据集（ACC + Audio）

  数据加载:
    - load_from_directory() — 从目录结构加载多用户数据
    - load_single_user_data() — 加载单用户数据
    - load_real_auth_data() — 加载真实多用户身份认证数据
    - create_dataloaders() — 创建训练/验证/测试 DataLoader
    - prepare_multi_user_simulation() — 从单用户数据模拟多用户（测试用）

  增强:
    - SpectrogramAugment — 频谱图在线数据增强
    - ContrastiveAugment — 对比学习专用增强（生成两个视图）
    - mixup_batch() — Batch-level Mixup 增强

数据格式：频谱图 (N, 80, 80)，归一化到 [-1, 1]
"""

import os
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 数据增强
# ============================================================

class SpectrogramAugment:
    """
    频谱图在线数据增强

    支持的增强操作：
    1. 时间遮蔽 (SpecAugment time mask)
    2. 频率遮蔽 (SpecAugment freq mask)
    3. 时间平移
    4. 高斯噪声
    5. 随机增益
    """

    def __init__(self,
                 time_mask_max=12,
                 freq_mask_max=10,
                 time_shift_max=8,
                 noise_std=0.03,
                 gain_range=(0.8, 1.2),
                 n_time_masks=2,
                 n_freq_masks=2,
                 p_time_mask=0.5,
                 p_freq_mask=0.5,
                 p_time_shift=0.3,
                 p_noise=0.4,
                 p_gain=0.3):
        self.time_mask_max  = time_mask_max
        self.freq_mask_max  = freq_mask_max
        self.time_shift_max = time_shift_max
        self.noise_std      = noise_std
        self.gain_range     = gain_range
        self.n_time_masks   = n_time_masks
        self.n_freq_masks   = n_freq_masks
        self.p_time_mask    = p_time_mask
        self.p_freq_mask    = p_freq_mask
        self.p_time_shift   = p_time_shift
        self.p_noise        = p_noise
        self.p_gain         = p_gain

    def time_mask(self, spec):
        """SpecAugment 时间遮蔽"""
        T = spec.shape[-1]
        for _ in range(self.n_time_masks):
            t_len = random.randint(1, min(self.time_mask_max, T - 1))
            t0 = random.randint(0, T - t_len)
            spec[..., t0:t0 + t_len] = 0
        return spec

    def freq_mask(self, spec):
        """SpecAugment 频率遮蔽"""
        F_bins = spec.shape[-2]
        for _ in range(self.n_freq_masks):
            f_len = random.randint(1, min(self.freq_mask_max, F_bins - 1))
            f0 = random.randint(0, F_bins - f_len)
            spec[..., f0:f0 + f_len, :] = 0
        return spec

    def time_shift(self, spec):
        """时间轴循环平移"""
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        return torch.roll(spec, shifts=shift, dims=-1)

    def add_noise(self, spec):
        """添加高斯噪声"""
        noise = torch.randn_like(spec) * self.noise_std
        return torch.clamp(spec + noise, -1, 1)

    def random_gain(self, spec):
        """随机增益缩放"""
        gain = random.uniform(*self.gain_range)
        return torch.clamp(spec * gain, -1, 1)

    def __call__(self, spec):
        """
        对单个频谱图应用随机增强组合

        Args:
            spec: (1, H, W) 或 (H, W) 的频谱图张量
        Returns:
            增强后的频谱图
        """
        spec = spec.clone()

        if random.random() < self.p_time_mask:
            spec = self.time_mask(spec)
        if random.random() < self.p_freq_mask:
            spec = self.freq_mask(spec)
        if random.random() < self.p_time_shift:
            spec = self.time_shift(spec)
        if random.random() < self.p_noise:
            spec = self.add_noise(spec)
        if random.random() < self.p_gain:
            spec = self.random_gain(spec)

        return spec


class ContrastiveAugment:
    """
    对比学习专用增强：为同一样本生成两个不同的增强视图

    用于 SimCLR 风格的自监督训练
    """

    def __init__(self, strength='medium'):
        if strength == 'light':
            self.aug = SpectrogramAugment(
                time_mask_max=8, freq_mask_max=6,
                time_shift_max=4, noise_std=0.02,
                n_time_masks=1, n_freq_masks=1,
                p_time_mask=0.4, p_freq_mask=0.4,
                p_time_shift=0.2, p_noise=0.3, p_gain=0.2
            )
        elif strength == 'medium':
            self.aug = SpectrogramAugment(
                time_mask_max=12, freq_mask_max=10,
                time_shift_max=8, noise_std=0.03,
                n_time_masks=2, n_freq_masks=2,
                p_time_mask=0.5, p_freq_mask=0.5,
                p_time_shift=0.3, p_noise=0.4, p_gain=0.3
            )
        elif strength == 'strong':
            self.aug = SpectrogramAugment(
                time_mask_max=16, freq_mask_max=14,
                time_shift_max=12, noise_std=0.05,
                n_time_masks=3, n_freq_masks=3,
                p_time_mask=0.7, p_freq_mask=0.7,
                p_time_shift=0.5, p_noise=0.5, p_gain=0.4
            )
        else:
            raise ValueError(f"Unknown strength: {strength}")

    def __call__(self, spec):
        """
        生成两个增强视图

        Args:
            spec: (1, H, W) 频谱图
        Returns:
            view1, view2: 两个不同的增强版本
        """
        return self.aug(spec), self.aug(spec)


def mixup_batch(specs, labels, alpha=0.2):
    """
    Batch-level Mixup 增强

    Args:
        specs:  (B, 1, H, W) 频谱图batch
        labels: (B,) 标签
        alpha:  Beta分布参数
    Returns:
        mixed_specs, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = specs.size(0)
    index = torch.randperm(batch_size, device=specs.device)

    mixed = lam * specs + (1 - lam) * specs[index]
    return mixed, labels, labels[index], lam


# ============================================================
# 2. 数据集
# ============================================================

class MultiUserSpectrogramDataset(Dataset):
    """
    多用户频谱图数据集

    每个样本返回: (spec, user_id)
    spec: (1, 80, 80) 频谱图
    user_id: int 用户标签
    """

    def __init__(self, specs, labels, augment=None, mode='acc'):
        """
        Args:
            specs:   (N, 80, 80) ndarray 或 list of ndarray
            labels:  (N,) ndarray 用户ID标签
            augment: SpectrogramAugment 实例（可选）
            mode:    'acc' 或 'audio'（标记数据类型）
        """
        self.specs   = torch.FloatTensor(specs)  # (N, 80, 80)
        self.labels  = torch.LongTensor(labels)  # (N,)
        self.augment = augment
        self.mode    = mode

        # 统计信息
        self.num_users   = len(torch.unique(self.labels))
        self.num_samples = len(self.specs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        spec  = self.specs[idx].unsqueeze(0)   # (1, 80, 80)
        label = self.labels[idx]

        if self.augment is not None:
            spec = self.augment(spec)

        return spec, label


class ContrastivePairDataset(Dataset):
    """
    对比学习配对数据集

    每个样本返回: (view1, view2, label)
    - view1, view2: 同一样本的两个不同增强视图
    - label: 用户ID

    用于 SimCLR / SupCon 训练
    """

    def __init__(self, specs, labels, augment_fn):
        """
        Args:
            specs:      (N, 80, 80) ndarray
            labels:     (N,) ndarray
            augment_fn: ContrastiveAugment 实例
        """
        self.specs      = torch.FloatTensor(specs)
        self.labels     = torch.LongTensor(labels)
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec  = self.specs[idx].unsqueeze(0)  # (1, 80, 80)
        label = self.labels[idx]

        view1, view2 = self.augment_fn(spec)
        return view1, view2, label


class DualModalDataset(Dataset):
    """
    双模态数据集：ACC + Audio 配对

    每个样本返回: (acc_spec, audio_spec, label)
    """

    def __init__(self, acc_specs, audio_specs, labels, augment=None):
        assert len(acc_specs) == len(audio_specs) == len(labels)
        self.acc_specs   = torch.FloatTensor(acc_specs)
        self.audio_specs = torch.FloatTensor(audio_specs)
        self.labels      = torch.LongTensor(labels)
        self.augment     = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        acc   = self.acc_specs[idx].unsqueeze(0)    # (1, 80, 80)
        audio = self.audio_specs[idx].unsqueeze(0)  # (1, 80, 80)
        label = self.labels[idx]

        if self.augment is not None:
            acc   = self.augment(acc)
            audio = self.augment(audio)

        return acc, audio, label


# ============================================================
# 3. 数据加载工具函数
# ============================================================

def load_from_directory(data_dir, modality='acc'):
    """
    从目录结构加载多用户数据

    目录结构：
        data_dir/
        ├── user_001/
        │   ├── acc_specs.npy    # (N1, 80, 80)
        │   └── audio_specs.npy  # (N1, 80, 80)
        ├── user_002/
        │   ├── acc_specs.npy
        │   └── audio_specs.npy
        └── ...

    Args:
        data_dir: 数据根目录
        modality: 'acc', 'audio', 或 'both'

    Returns:
        specs: (N_total, 80, 80) 或 (acc_specs, audio_specs)
        labels: (N_total,)
        user_names: list of str
    """
    all_specs_acc   = []
    all_specs_audio = []
    all_labels      = []
    user_names      = []

    user_dirs = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])

    for user_id, user_dir in enumerate(user_dirs):
        user_path = os.path.join(data_dir, user_dir)
        user_names.append(user_dir)

        if modality in ('acc', 'both'):
            acc_path = os.path.join(user_path, 'acc_specs.npy')
            if os.path.exists(acc_path):
                acc_data = np.load(acc_path)
                all_specs_acc.append(acc_data)

        if modality in ('audio', 'both'):
            audio_path = os.path.join(user_path, 'audio_specs.npy')
            if os.path.exists(audio_path):
                audio_data = np.load(audio_path)
                all_specs_audio.append(audio_data)

        # 确定样本数
        if all_specs_acc:
            n = len(all_specs_acc[-1])
        elif all_specs_audio:
            n = len(all_specs_audio[-1])
        else:
            continue

        all_labels.extend([user_id] * n)

    labels = np.array(all_labels)

    if modality == 'acc':
        return np.concatenate(all_specs_acc, axis=0), labels, user_names
    elif modality == 'audio':
        return np.concatenate(all_specs_audio, axis=0), labels, user_names
    else:  # both
        return (np.concatenate(all_specs_acc, axis=0),
                np.concatenate(all_specs_audio, axis=0),
                labels, user_names)


def load_single_user_data(acc_path, audio_path=None, user_id=0):
    """
    加载单用户数据（当前 bo 数据的便捷接口）

    Args:
        acc_path:   ACC频谱图路径 (.npy)
        audio_path: Audio频谱图路径 (.npy)，可选
        user_id:    用户ID标签

    Returns:
        acc_specs, (audio_specs), labels
    """
    acc_specs = np.load(acc_path)  # (N, 80, 80)
    labels = np.full(len(acc_specs), user_id, dtype=np.int64)

    if audio_path is not None:
        audio_specs = np.load(audio_path)
        return acc_specs, audio_specs, labels

    return acc_specs, labels


# def load_real_auth_data():
#     """
#     加载真实多用户身份认证数据

#     用户0 (bo):  ./mei/1/acc_specs.npy — CycleGAN val+test (96样本)
#     用户1 (mei): ./mei/2/acc_specs.npy — 全部
#     用户2:       ./mei/3/acc_specs.npy — 全部
#     用户3:       ./mei/4/acc_specs.npy — 全部

#     Returns:
#         all_specs: (N_total, 80, 80)
#         all_labels: (N_total,)
#         user_names: list of str
#     """
#     all_specs_list = []
#     all_labels_list = []
#     user_names = []

#     # ---- 用户0: bo (CycleGAN val+test) ----
#     bo_path = './meiy/1/acc.npy'
#     bo_all = np.load(bo_path)  # (300, 80, 80)

#     # 复现CycleGAN训练时的数据划分 (random_state=42, test=20%, val=20%)
#     indices = np.arange(len(bo_all))
#     train_val_idx, test_idx = train_test_split(
#         indices, test_size=0.2, random_state=42)
#     val_ratio = 0.2 / (1 - 0.2)
#     train_idx, val_idx = train_test_split(
#         train_val_idx, test_size=val_ratio, random_state=42)

#     # 只取验证集+测试集（CycleGAN没见过的数据）
#     auth_idx = np.concatenate([val_idx, test_idx])
#     bo_specs = bo_all[auth_idx]  # (96, 80, 80)

#     # 统一裁剪到80x80（bo数据可能是80x81）
#     if bo_specs.shape[1] > 80:
#         bo_specs = bo_specs[:, :80, :]
#     if bo_specs.shape[2] > 80:
#         bo_specs = bo_specs[:, :, :80]

#     all_specs_list.append(bo_specs)
#     all_labels_list.append(np.zeros(len(bo_specs), dtype=np.int64))
#     user_names.append('bo')

#     print(f"  用户0 (bo): {len(bo_specs)} 样本 (CycleGAN val+test)")
#     print(f"    来源: {bo_path}")
#     print(f"    val={len(val_idx)}, test={len(test_idx)}")

#     # ---- 用户1~3: 其他用户，全部样本 ----
#     other_users = [
#         ('uesr2', './meiy/2/acc.npy'),
#         ('user3', './meiy/3/acc.npy'),
#         # ('user4', './meiy/4/acc_specs.npy'),
#         ('user5', './meiy/5/acc.npy'),
#         ('user6', './meiy/6/acc.npy'),
#         ('user7', './meiy/7/acc.npy'),
#         ('user8', './meiy/8/acc.npy'),
#         ('user9', './meiy/9/acc.npy'),
#         # ('user3', './meiy/3/acc_specs.npy'),
#         # # ('user4', './meiy/4/acc_specs.npy'),
#         # ('user5', './meiy/5/acc_specs.npy'),
#         # ('user6', './meiy/6/acc_specs.npy'),
#         # ('user7', './meiy/7/acc_specs.npy'),
#         # ('user8', './meiy/8/acc_specs.npy'),
#         # ('user9', './meiy/9/acc_specs.npy'),
        
#     ]
#     for uid, (name, path) in enumerate(other_users, start=1):
#         specs = np.load(path)
#         # 统一裁剪到80x80（某些用户数据可能是80x81等）
#         if specs.shape[1] > 80:
#             specs = specs[:, :80, :]
#         if specs.shape[2] > 80:
#             specs = specs[:, :, :80]
#         all_specs_list.append(specs)
#         all_labels_list.append(np.full(len(specs), uid, dtype=np.int64))
#         user_names.append(name)
#         print(f"  用户{uid} ({name}): {len(specs)} 样本")
#         print(f"    来源: {path}")

#     # ---- 合并 ----
#     all_specs = np.concatenate(all_specs_list, axis=0)
#     all_labels = np.concatenate(all_labels_list, axis=0)

#     print(f"  合计: {len(all_specs)} 样本, {len(user_names)} 用户")
#     counts = {name: int(np.sum(all_labels == i)) for i, name in enumerate(user_names)}
#     print(f"  各用户样本数: {counts}")

#     return all_specs, all_labels, user_names
def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cyclegan_split_indices(n_samples, test_size=0.1, val_size=0.1,
                            random_state=42):
    """Reproduce DataLoader._split_data without loading sample contents."""
    indices = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state)
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size,
        random_state=random_state)
    return train_idx, val_idx, test_idx


def load_real_auth_data(protocol='all', cyclegan_test_size=0.1,
                        cyclegan_val_size=0.1, cyclegan_seed=42,
                        return_metadata=False):
    """
    加载真实多用户身份认证数据。

    protocol='all' 保留旧行为；protocol='cyclegan_holdout' 只加载与历史
    CycleGAN 完全相同划分下的 validation + test 样本。后者仍需由
    create_dataloaders 再划分认证头 train/validation/test。
    """
    all_specs_list = []
    all_labels_list = []
    user_names = []
    all_sample_ids = []
    source_manifest = {
        'protocol': protocol,
        'cyclegan_split': {
            'test_size': cyclegan_test_size,
            'val_size': cyclegan_val_size,
            'random_state': cyclegan_seed,
        },
        'users': {},
    }

    # ---- 用户0: bo (加载全部数据) ----
    bo_path = './meiy/1/acc.npy'
    bo_all = np.load(bo_path)  # (300, 80, 80)
    
    if protocol == 'cyclegan_holdout':
        bo_train_idx, bo_val_idx, bo_test_idx = _cyclegan_split_indices(
            len(bo_all), cyclegan_test_size, cyclegan_val_size,
            cyclegan_seed)
        bo_auth_idx = np.concatenate([bo_val_idx, bo_test_idx])
        bo_specs = bo_all[bo_auth_idx]
    elif protocol == 'all':
        bo_train_idx = np.array([], dtype=np.int64)
        bo_val_idx = np.array([], dtype=np.int64)
        bo_test_idx = np.array([], dtype=np.int64)
        bo_auth_idx = np.arange(len(bo_all))
        bo_specs = bo_all
    else:
        raise ValueError(f"Unknown authentication data protocol: {protocol}")

    # 统一裁剪到80x80（bo数据可能是80x81）
    if bo_specs.shape[1] > 80:
        bo_specs = bo_specs[:, :80, :]
    if bo_specs.shape[2] > 80:
        bo_specs = bo_specs[:, :, :80]

    all_specs_list.append(bo_specs)
    all_labels_list.append(np.zeros(len(bo_specs), dtype=np.int64))
    all_sample_ids.extend([f'bo:{int(i)}' for i in bo_auth_idx])
    user_names.append('bo')

    source_manifest['users']['bo'] = {
        'acc_path': bo_path,
        'acc_sha256': _sha256_file(bo_path),
        'n_total': len(bo_all),
        'n_cyclegan_train': len(bo_train_idx),
        'n_cyclegan_val': len(bo_val_idx),
        'n_cyclegan_test': len(bo_test_idx),
        'selected_indices': bo_auth_idx.tolist(),
    }
    print(f"  用户0 (bo): {len(bo_specs)} 样本 (protocol={protocol})")
    print(f"    来源: {bo_path}")

    # ---- 用户1~N: 其他用户，全部样本 ----
    other_users = [
        ('user2', './meiy/2/acc.npy'),  # 修复了原拼写错误 'uesr2'
        ('user3', './meiy/3/acc.npy'),
        ('user4', './meiy/4/acc.npy'),
        ('user5', './meiy/5/acc.npy'),
        ('user6', './meiy/6/acc.npy'),
        ('user7', './meiy/7/acc.npy'),
        ('user8', './meiy/8/acc.npy'),
        ('user9', './meiy/9/acc.npy'),
    ]
    
    for uid, (name, path) in enumerate(other_users, start=1):
        specs = np.load(path)
        if protocol == 'cyclegan_holdout':
            train_idx, val_idx, test_idx = _cyclegan_split_indices(
                len(specs), cyclegan_test_size, cyclegan_val_size,
                cyclegan_seed)
            auth_idx = np.concatenate([val_idx, test_idx])
            selected_specs = specs[auth_idx]
        else:
            train_idx = np.array([], dtype=np.int64)
            val_idx = np.array([], dtype=np.int64)
            test_idx = np.array([], dtype=np.int64)
            auth_idx = np.arange(len(specs))
            selected_specs = specs
        # 统一裁剪到80x80
        if selected_specs.shape[1] > 80:
            selected_specs = selected_specs[:, :80, :]
        if selected_specs.shape[2] > 80:
            selected_specs = selected_specs[:, :, :80]

        all_specs_list.append(selected_specs)
        all_labels_list.append(np.full(len(selected_specs), uid, dtype=np.int64))
        all_sample_ids.extend([f'{name}:{int(i)}' for i in auth_idx])
        user_names.append(name)

        source_manifest['users'][name] = {
            'acc_path': path,
            'acc_sha256': _sha256_file(path),
            'n_total': len(specs),
            'n_cyclegan_train': len(train_idx),
            'n_cyclegan_val': len(val_idx),
            'n_cyclegan_test': len(test_idx),
            'selected_indices': auth_idx.tolist(),
        }

        print(f"  用户{uid} ({name}): {len(selected_specs)} 样本")
        print(f"    来源: {path}")

    # ---- 合并 ----
    all_specs = np.concatenate(all_specs_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    print(f"\n  合计加载: {len(all_specs)} 样本, {len(user_names)} 用户")
    counts = {name: int(np.sum(all_labels == i)) for i, name in enumerate(user_names)}
    print(f"  各用户样本数分布: {counts}\n")

    if protocol == 'cyclegan_holdout':
        selected = set(all_sample_ids)
        cyclegan_train = {
            f'{name}:{idx}'
            for name, item in source_manifest['users'].items()
            for idx in range(item['n_total'])
            if idx not in set(item['selected_indices'])
        }
        overlap = selected.intersection(cyclegan_train)
        if overlap:
            raise RuntimeError(
                f'CycleGAN train/authentication pool overlap: {len(overlap)}')
        source_manifest['assertions'] = {
            'cyclegan_train_auth_pool_overlap': 0,
            'passed': True,
        }

    if return_metadata:
        return (all_specs, all_labels, user_names,
                np.asarray(all_sample_ids), source_manifest)
    return all_specs, all_labels, user_names

def create_dataloaders(specs, labels,
                       batch_size=32,
                       test_size=0.2, val_size=0.2,
                       augment=None,
                       contrastive_augment=None,
                       random_state=42,
                       num_workers=0,
                       sample_ids=None,
                       split_manifest_path=None,
                       manifest_extra=None):
    """
    创建训练/验证/测试 DataLoader

    Args:
        specs:   (N, 80, 80) 频谱图数据
        labels:  (N,) 用户标签
        batch_size: batch大小
        test_size:  测试集比例
        val_size:   验证集比例（从训练集中分出）
        augment:    训练集增强
        contrastive_augment: 对比增强（如果提供，使用ContrastivePairDataset）
        random_state: 随机种子

    Returns:
        train_loader, val_loader, test_loader
    """
    # 划分数据
    if sample_ids is None:
        sample_ids = np.asarray([str(i) for i in range(len(specs))])
    else:
        sample_ids = np.asarray(sample_ids)
    if len(sample_ids) != len(specs):
        raise ValueError('sample_ids and specs must have equal length')

    (train_val_specs, test_specs, train_val_labels, test_labels,
     train_val_ids, test_ids) = train_test_split(
        specs, labels, sample_ids, test_size=test_size,
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None)

    val_ratio = val_size / (1 - test_size)
    (train_specs, val_specs, train_labels, val_labels,
     train_ids, val_ids) = train_test_split(
        train_val_specs, train_val_labels, train_val_ids,
        test_size=val_ratio, random_state=random_state,
        stratify=train_val_labels if len(np.unique(train_val_labels)) > 1 else None)

    split_sets = [set(train_ids), set(val_ids), set(test_ids)]
    if (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2]
            or split_sets[1] & split_sets[2]):
        raise RuntimeError('Authentication train/validation/test overlap')

    if split_manifest_path:
        manifest = {
            'random_state': random_state,
            'test_size': test_size,
            'val_size': val_size,
            'counts': {
                'train': len(train_ids),
                'validation': len(val_ids),
                'test': len(test_ids),
            },
            'sample_ids': {
                'train': train_ids.tolist(),
                'validation': val_ids.tolist(),
                'test': test_ids.tolist(),
            },
            'assertions': {
                'authentication_split_overlap': 0,
                'passed': True,
            },
        }
        if manifest_extra is not None:
            manifest['source'] = manifest_extra
        parent = os.path.dirname(split_manifest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(split_manifest_path, 'w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        print(f"划分清单: {split_manifest_path}")

    print(f"数据划分: 训练={len(train_specs)}, 验证={len(val_specs)}, 测试={len(test_specs)}")
    print(f"用户分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 创建Dataset
    if contrastive_augment is not None:
        train_dataset = ContrastivePairDataset(train_specs, train_labels, contrastive_augment)
    else:
        train_dataset = MultiUserSpectrogramDataset(train_specs, train_labels, augment=augment)

    val_dataset  = MultiUserSpectrogramDataset(val_specs,  val_labels,  augment=None)
    test_dataset = MultiUserSpectrogramDataset(test_specs, test_labels, augment=None)

    # 创建DataLoader
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader   = TorchDataLoader(val_dataset,   batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    test_loader  = TorchDataLoader(test_dataset,  batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def prepare_multi_user_simulation(acc_specs, n_fake_users=4,
                                   noise_level=0.15, random_state=42):
    """
    从单用户数据模拟多用户（用于测试框架）
    注意：这只是用于测试代码流程，不代表真实的多用户场景！
    """
    rng = np.random.RandomState(random_state)
    all_specs  = [acc_specs.copy()]
    all_labels = [np.zeros(len(acc_specs), dtype=np.int64)]
    for uid in range(1, n_fake_users + 1):
        fake = acc_specs.copy()
        fake = fake + rng.uniform(-0.2, 0.2)
        fake = fake * rng.uniform(0.85, 1.15, size=(1, 80, 1)).astype(np.float32)
        fake = fake + rng.randn(*fake.shape).astype(np.float32) * noise_level
        fake = np.clip(fake, -1, 1)
        all_specs.append(fake)
        all_labels.append(np.full(len(fake), uid, dtype=np.int64))
    return np.concatenate(all_specs, axis=0), np.concatenate(all_labels, axis=0)
