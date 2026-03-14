import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, acc_path, audio_path, test_size=0.2, val_size=0.1, random_state=42, 
                 normalize=True, standardize=False):
        """
        数据加载器初始化，支持训练/验证/测试三划分
        
        :param acc_path: 加速度数据文件路径（npy 文件）
        :param audio_path: 音频数据文件路径（npy 文件）
        :param test_size: 测试集比例
        :param val_size: 验证集比例
        :param random_state: 随机种子，用于可重复性
        :param normalize: 是否对数据进行归一化
        :param standardize: 是否对数据进行标准化
        """
        self.acc_path = acc_path
        self.audio_path = audio_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.normalize = None
        self.standardize = None  
        # 新添加的数据存储
        self.val_acc = None
        self.val_audio = None
        self.acc_data = None
        self.audio_data = None
        self.train_acc = None
        self.train_audio = None
        self.test_acc = None
        self.test_audio = None
        self.acc_mean = None
        self.acc_std = None
        
        self._load_data()
        self._preprocess_data()
        self._split_data()
        self._print_data_stats()  # 打印数据集统计信息

    def _load_data(self):
        """加载加速度和音频数据"""
        self.acc_data = np.load(self.acc_path)
        self.audio_data = np.load(self.audio_path)
        print(f"加载完成: 加速度数据 shape {self.acc_data.shape}, 音频数据 shape {self.audio_data.shape}")

    def _preprocess_data(self):
        """数据预处理：归一化或标准化"""
        if self.normalize:
            # 归一化到 [0, 1]
            acc_min = self.acc_data.min()
            acc_max = self.acc_data.max()
            self.acc_data = (self.acc_data - acc_min) / (acc_max - acc_min)
            print(f"加速度数据归一化完成: min={acc_min:.4f}, max={acc_max:.4f}")

        if self.standardize:
            # 标准化到均值 0，标准差 1
            self.acc_mean = self.acc_data.mean()
            self.acc_std = self.acc_data.std()
            self.acc_data = (self.acc_data - self.acc_mean) / self.acc_std
            print(f"加速度数据标准化完成: mean={self.acc_mean:.4f}, std={self.acc_std:.4f}")

    def _split_data(self):
        """划分训练集、验证集和测试集（6:2:2比例）"""
        # 首先划分训练+验证与测试集 (80%训练验证, 20%测试)
        total_indices = np.arange(self.acc_data.shape[0])
        train_val_acc, self.test_acc, train_val_audio, self.test_audio, train_val_idx, test_idx = train_test_split(
            self.acc_data, self.audio_data, total_indices, 
            test_size=self.test_size, random_state=self.random_state
        )
        
        # 然后从训练验证集中再划分训练和验证 (60%训练, 20%验证)
        # 注意: 这里val_size相对于剩余数据的比例是20%/80%=25%
        relative_val_size = self.val_size / (1 - self.test_size)
        self.train_acc, self.val_acc, self.train_audio, self.val_audio, _, val_idx = train_test_split(
            train_val_acc, train_val_audio, train_val_idx,
            test_size=relative_val_size, random_state=self.random_state
        )
        
        # 打印划分结果
        n_total = self.acc_data.shape[0]
        print(f"数据划分完成 (总样本数={n_total}):")
        print(f"  训练集: {len(self.train_acc)}样本 ({len(self.train_acc)/n_total:.1%})")
        print(f"  验证集: {len(self.val_acc)}样本 ({len(self.val_acc)/n_total:.1%})")
        print(f"  测试集: {len(self.test_acc)}样本 ({len(self.test_acc)/n_total:.1%})")
    
    def _print_data_stats(self):
        """打印各数据集统计信息"""
        print("\n数据集统计信息:")
        print(f"训练集 - 加速度: min={self.train_acc.min():.4f}, max={self.train_acc.max():.4f}, mean={self.train_acc.mean():.4f}")
        print(f"验证集 - 加速度: min={self.val_acc.min():.4f}, max={self.val_acc.max():.4f}, mean={self.val_acc.mean():.4f}")
        print(f"测试集 - 加速度: min={self.test_acc.min():.4f}, max={self.test_acc.max():.4f}, mean={self.test_acc.mean():.4f}")
        print(f"训练集 - 音频: min={self.train_audio.min():.4f}, max={self.train_audio.max():.4f}, mean={self.train_audio.mean():.4f}")
        print(f"验证集 - 音频: min={self.val_audio.min():.4f}, max={self.val_audio.max():.4f}, mean={self.val_audio.mean():.4f}")
        print(f"测试集 - 音频: min={self.test_audio.min():.4f}, max={self.test_audio.max():.4f}, mean={self.test_audio.mean():.4f}")
    def get_batch(self, batch_size, dataset='train', indices=None):
        """
        按段提取数据样本，支持批量加载，可选特定样本索引
        
        :param batch_size: 每批加载的样本数量
        :param dataset: 指定数据集，'train', 'val' 或 'test'
        :param indices: 具体选择的样本索引列表，默认为 None（随机选择）
        :return: (加速度样本, 音频样本)
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
            # 随机选择样本
            indices = np.random.choice(total_samples, size=batch_size, replace=False)
        else:
            # 使用指定的索引
            indices = np.array(indices)
            if len(indices) != batch_size:
                raise ValueError(f"索引长度 {len(indices)} 与批次大小 {batch_size} 不匹配")
            if np.any(indices >= total_samples) or np.any(indices < 0):
                raise ValueError("部分索引超出范围")

        acc_samples = acc_data[indices]
        audio_samples = audio_data[indices]

        return acc_samples, audio_samples

    def get_batch_iter(self, batch_size, dataset='train'):
        """
        返回一个批次迭代器，逐批次提供数据
        
        :param batch_size: 每批的样本数量
        :param dataset: 指定数据集（'train', 'val' 或 'test'）
        :return: 生成器对象
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
        num_batches = total_samples // batch_size
        
        # 最后一批可能不满，所以使用range+切片
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield acc_data[start_idx:end_idx], audio_data[start_idx:end_idx]
        
        # 如果最后一批还有剩余样本
        if total_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            yield acc_data[start_idx:], audio_data[start_idx:]

# 测试代码
if __name__ == "__main__":
    # 测试DataLoader
    data_loader = DataLoader(
        acc_path="<你的加速度数据路径>",
        audio_path="<你的音频数据路径>",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        normalize=True,
        standardize=False
    )
    
    # 测试获取训练数据
    train_batch = data_loader.get_batch(batch_size=16, dataset='train')
    print(f"训练批次形状: acc={train_batch[0].shape}, audio={train_batch[1].shape}")
    
    # 测试验证集迭代器
    print("\n验证集批次迭代:")
    for i, (val_acc, val_audio) in enumerate(data_loader.get_batch_iter(batch_size=32, dataset='val')):
        print(f"批次 {i+1}: acc={val_acc.shape}, audio={val_audio.shape}")
    
    # 测试测试集迭代器
    test_iter = data_loader.get_batch_iter(batch_size=64, dataset='test')
    test_batch = next(iter(test_iter))
    print(f"\n测试批次形状: acc={test_batch[0].shape}, audio={test_batch[1].shape}")