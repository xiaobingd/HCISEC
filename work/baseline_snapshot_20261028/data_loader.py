import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, acc_path, audio_path, test_size=0.2, val_size=0.2, random_state=42):
        """
        数据加载器 - 适配预处理后的频谱图数据
        
        输入数据已经是频谱图格式 (N, 80, 81)，归一化到 [-1, 1]
        - ACC: STFT频谱图 (N, 80, 81)
        - Audio: Mel频谱图 (N, 80, 81)
        
        :param acc_path: ACC STFT频谱图数据路径 (npy)
        :param audio_path: Audio Mel频谱图数据路径 (npy)
        :param test_size: 测试集比例
        :param val_size: 验证集比例
        :param random_state: 随机种子
        """
        self.acc_path = acc_path
        self.audio_path = audio_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.val_acc = None
        self.val_audio = None
        self.acc_data = None
        self.audio_data = None
        self.train_acc = None
        self.train_audio = None
        self.test_acc = None
        self.test_audio = None

        self._load_data()
        self._split_data()
        self._print_data_stats()

    def _load_data(self):
        """加载预处理后的频谱图数据"""
        self.acc_data = np.load(self.acc_path)
        self.audio_data = np.load(self.audio_path)
        print(f"加载完成: ACC数据 shape {self.acc_data.shape}, Audio数据 shape {self.audio_data.shape}")
        print(f"  ACC  范围: [{self.acc_data.min():.3f}, {self.acc_data.max():.3f}]")
        print(f"  Audio范围: [{self.audio_data.min():.3f}, {self.audio_data.max():.3f}]")

    def _split_data(self):
        """划分训练集、验证集和测试集（6:2:2比例）"""
        total_indices = np.arange(self.acc_data.shape[0])
        train_val_acc, self.test_acc, train_val_audio, self.test_audio, train_val_idx, test_idx = train_test_split(
            self.acc_data, self.audio_data, total_indices,
            test_size=self.test_size, random_state=self.random_state
        )

        relative_val_size = self.val_size / (1 - self.test_size)
        self.train_acc, self.val_acc, self.train_audio, self.val_audio, _, val_idx = train_test_split(
            train_val_acc, train_val_audio, train_val_idx,
            test_size=relative_val_size, random_state=self.random_state
        )

        n_total = self.acc_data.shape[0]
        print(f"数据划分完成 (总样本数={n_total}):")
        print(f"  训练集: {len(self.train_acc)}样本 ({len(self.train_acc)/n_total:.1%})")
        print(f"  验证集: {len(self.val_acc)}样本 ({len(self.val_acc)/n_total:.1%})")
        print(f"  测试集: {len(self.test_acc)}样本 ({len(self.test_acc)/n_total:.1%})")

    def _print_data_stats(self):
        """打印各数据集统计信息"""
        print("\n数据集统计信息:")
        for name, acc, audio in [
            ("训练集", self.train_acc, self.train_audio),
            ("验证集", self.val_acc, self.val_audio),
            ("测试集", self.test_acc, self.test_audio),
        ]:
            print(f"{name} - ACC:   min={acc.min():.4f}, max={acc.max():.4f}, mean={acc.mean():.4f}")
            print(f"{name} - Audio: min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")

    def get_batch(self, batch_size, dataset='train', indices=None):
        """
        按段提取数据样本
        返回: (acc_batch, audio_batch)，shape 均为 (B, 80, 81)
        """
        if dataset == 'train':
            acc_data = self.train_acc
            audio_data = self.train_audio
        elif dataset == 'val':
            acc_data = self.val_acc
            audio_data = self.val_audio
        elif dataset == 'test':
            acc_data = self.test_acc
            audio_data = self.test_audio
        else:
            raise ValueError("Dataset must be 'train', 'val', or 'test'.")

        total_samples = acc_data.shape[0]

        if indices is None:
            indices = np.random.choice(total_samples, size=batch_size, replace=False)
        else:
            indices = np.array(indices)
            if len(indices) != batch_size:
                raise ValueError(f"索引长度 {len(indices)} 与批次大小 {batch_size} 不匹配")
            if np.any(indices >= total_samples) or np.any(indices < 0):
                raise ValueError("部分索引超出范围")

        return acc_data[indices], audio_data[indices]

    def get_batch_iter(self, batch_size, dataset='train', shuffle=True):
        """
        返回一个批次迭代器，逐批次提供数据
        
        :param batch_size: 每批的样本数量
        :param dataset: 指定数据集
        :param shuffle: 是否打乱顺序（训练集建议True）
        :return: 生成器对象，yield (acc_batch, audio_batch)
        """
        if dataset == 'train':
            acc_data = self.train_acc
            audio_data = self.train_audio
        elif dataset == 'val':
            acc_data = self.val_acc
            audio_data = self.val_audio
        elif dataset == 'test':
            acc_data = self.test_acc
            audio_data = self.test_audio
        elif dataset == 'all':
            acc_data = self.acc_data
            audio_data = self.audio_data
        else:
            raise ValueError("Dataset must be 'train', 'val', 'test' or 'all'.")

        total_samples = acc_data.shape[0]
        
        if shuffle:
            perm = np.random.permutation(total_samples)
            acc_data = acc_data[perm]
            audio_data = audio_data[perm]
        
        num_batches = total_samples // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield acc_data[start_idx:end_idx], audio_data[start_idx:end_idx]

        # 最后一批剩余样本
        if total_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            yield acc_data[start_idx:], audio_data[start_idx:]


# 测试代码
if __name__ == "__main__":
    import os
    data_dir = "/home/admin/workspace/downloads/analysis_output"
    
    data_loader = DataLoader(
        acc_path=os.path.join(data_dir, "cyclegan_acc_selected.npy"),
        audio_path=os.path.join(data_dir, "cyclegan_audio_selected.npy"),
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    # 测试获取训练数据
    train_batch = data_loader.get_batch(batch_size=16, dataset='train')
    print(f"\n训练批次形状: acc={train_batch[0].shape}, audio={train_batch[1].shape}")

    # 测试迭代器
    print("\n训练集批次迭代:")
    for i, (acc, audio) in enumerate(data_loader.get_batch_iter(batch_size=16, dataset='train')):
        print(f"  批次 {i+1}: acc={acc.shape}, audio={audio.shape}")
