import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from scipy.signal import stft, butter, filtfilt
import torchaudio
import torch.autograd as autograd


# 可视化函数 (使用colorize)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples.detach()).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=False,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
def dynamic_normalize_log(tensor, epsilon=1e-6):
    """
    对张量进行对数变换，然后再进行独立的动态归一化。
    非常适合高动态范围的图像，如STFT频谱图。
    """
    # 加上一个很小的数，避免log(0)导致无穷大
    log_tensor = torch.log(tensor + epsilon)
    
    min_val = log_tensor.min()
    max_val = log_tensor.max()
    # 避免除以零的错误
    if max_val > min_val:
        return (log_tensor - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(tensor)
def compute_mel_spectrogram(audio_batch, sample_rate=10448, n_fft=512, hop_length=256, n_mels=64):
    """
    计算音频的梅尔频谱
    :param audio_batch: 音频数据 [batch_size, samples] 或 [batch_size, samples, 1]
    :param sample_rate: 采样率
    :param n_fft: FFT窗口大小
    :param hop_length: 帧移
    :param n_mels: 梅尔滤波器数量
    :return: 梅尔频谱 [batch_size, 1, n_mels, time]
    """
    if audio_batch.shape[-1] == 1:
        audio_batch = audio_batch.squeeze(-1)

    if isinstance(audio_batch, np.ndarray):
        audio_batch = torch.from_numpy(audio_batch).float()

    device = audio_batch.device
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(device)

    mel = mel_transform(audio_batch)  # [batch_size, n_mels, time]
    return mel.unsqueeze(1)  # [batch_size, 1, n_mels, time]


# def normalize_mel_spectrogram(mel_spectrogram):
#     """
#     对梅尔频谱进行归一化
#     :param mel_spectrogram: 梅尔频谱 [batch_size, 1, n_mels, time]
#     :return: 归一化后的梅尔频谱
#     """
#     log_mel = torch.log(mel_spectrogram + 1e-9)
#     min_val, _ = log_mel.min(dim=-1, keepdim=True)
#     max_val, _ = log_mel.max(dim=-1, keepdim=True)
#     return 2*(log_mel - min_val) / (max_val - min_val + 1e-6)-1


def normalize_mel_spectrogram(mel_spectrogram):
    
    # 动态获取每个样本的 min 和 max
    # 假设输入形状 [B, 1, F, T] 或 [1, F, T]
    # 我们希望对每张图独立归一化
    if mel_spectrogram.dim() == 3: # [1, F, T]
        min_val = mel_spectrogram.min()
        max_val = mel_spectrogram.max()
    elif img.dim() == 4: # [B, 1, F, T]
        min_val = mel_spectrogram.view(img.size(0), -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_val = mel_spectrogram.view(img.size(0), -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    
    # 归一化到 [-1, 1]
    # 加上 1e-6 防止 max == min 导致除以 0
    norm_img =  (mel_spectrogram - min_val) / (max_val - min_val + 1e-6) 
    return norm_img

def normalize_magnitude(magnitude):
    """
    归一化幅度谱
    :param magnitude: 幅度谱
    :return: 归一化后的幅度谱
    """
    min_val = magnitude.min()
    max_val = magnitude.max()
    return (magnitude - min_val) / (max_val - min_val + 1e-6)

def denormalize_magnitude(normalized_mag, original_mag):
    """
    将归一化后的幅度谱还原为原始范围
    :param normalized_mag: 归一化后的谱 (通常是模型输出)
    :param original_mag: 原始未归一化的谱 (如训练数据输入时的 mag)
    :return: 反归一化后的谱
    """
    min_val = original_mag.min()
    max_val = original_mag.max()
    return normalized_mag * (max_val - min_val + 1e-6) + min_val

def extract_stft_features(signal_batch, n_fft=512, hop_length=256, win_length=512):
    """
    提取STFT特征（幅度谱和相位谱）
    :param signal_batch: 信号数据 [batch_size, channels, samples] 或 [batch_size, samples, channels]
    :param n_fft: FFT窗口大小
    :param hop_length: 帧移
    :param win_length: 窗口长度
    :return: 幅度谱和相位谱 ([batch_size, channels, freq, time], [batch_size, channels, freq, time])
    """
    # 确保输入是[batch_size, channels, samples]格式
    if signal_batch.shape[-1] <= 3:  # 通道维度在最后
        signal_batch = signal_batch.transpose(0, 2, 1)

    if isinstance(signal_batch, np.ndarray):
        signal_batch = torch.from_numpy(signal_batch).float()

    device = signal_batch.device
    batch_size, channels, samples = signal_batch.shape

    # 创建汉宁窗
    window = torch.hann_window(win_length, device=device)

    # 存储结果
    magnitudes = []
    phases = []

    # 对每个通道进行STFT
    for c in range(channels):
        stft = torch.stft(
            signal_batch[:, c],  # [batch_size, samples]
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        # 提取幅度和相位
        magnitude = torch.abs(stft)  # [batch_size, freq, time]
        phase = torch.angle(stft)  # [batch_size, freq, time]

        magnitudes.append(magnitude.unsqueeze(1))  # [batch_size, 1, freq, time]
        phases.append(phase.unsqueeze(1))  # [batch_size, 1, freq, time]

    # 连接所有通道
    if channels > 1:
        magnitudes = torch.cat(magnitudes, dim=1)  # [batch_size, channels, freq, time]
        phases = torch.cat(phases, dim=1)  # [batch_size, channels, freq, time]
    else:
        magnitudes = magnitudes[0]  # [batch_size, 1, freq, time]
        phases = phases[0]  # [batch_size, 1, freq, time]

    return magnitudes, phases


def istft(magnitude, phase, hop_length=256, win_length=512, n_fft=512):
    """
    逆短时傅里叶变换，将频域转回时域
    :param magnitude: 幅度谱 [batch_size, freq, time]
    :param phase: 相位谱 [batch_size, freq, time]
    :param hop_length: 帧移
    :param win_length: 窗口长度
    :param n_fft: FFT窗口大小
    :return: 重建的时域信号 [batch_size, samples]
    """
    if isinstance(magnitude, np.ndarray):
        magnitude = torch.from_numpy(magnitude).float()
    if isinstance(phase, np.ndarray):
        phase = torch.from_numpy(phase).float()

    device = magnitude.device

    # 转回复数域
    complex_spec = magnitude * torch.exp(1j * phase)

    # 创建汉宁窗
    window = torch.hann_window(win_length, device=device)

    # 执行 iSTFT
    signal = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False
    )

    return signal

def minmax_fullimage(x): # x:[B,C,F,T]
    x = torch.log(x + 1e-9)
    mn = x.amin(dim=(1,2,3), keepdim=True)
    mx = x.amax(dim=(1,2,3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-6)

def plot_mel_spectrogram(mel_spectrogram, title='Mel Spectrogram'):
    """
    绘制梅尔频谱图
    :param mel_spectrogram: 梅尔频谱 [batch_size, 1, n_mels, time] 或 [n_mels, time]
    :param title: 图表标题
    """
    if len(mel_spectrogram.shape) == 4:
        # 取第一个样本
        mel_spectrogram = mel_spectrogram[0, 0]
    elif len(mel_spectrogram.shape) == 3:
        # 取第一个通道
        mel_spectrogram = mel_spectrogram[0]

    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.detach().cpu().numpy()

    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency (Hz)')



def plot_magnitude_spectrum(mag, title=None, fs=10448, n_fft=512):

    mag = mag.squeeze().cpu().detach().numpy()

    num_freq_bins, num_time_frames = mag.shape

    # 频率坐标 (只保留 0 ~ fs/2 部分）
    freq_axis = np.linspace(0, fs / 2, num_freq_bins)

    # 时间坐标
    hop_length = n_fft // 2  # 默认50%重叠
    duration = num_time_frames * hop_length / fs
    time_axis = np.linspace(0, duration, num_time_frames)

    plt.imshow(mag, aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.colorbar(label="Normalized Magnitude")
    if title is not None:
        plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0,200)
    plt.tight_layout()


def plot_waveform(waveform, sample_rate=10448, title='Waveform'):
    """
    绘制波形图
    :param waveform: 波形数据 [batch_size, samples] 或 [samples]
    :param sample_rate: 采样率
    :param title: 图表标题
    """
    if len(waveform.shape) > 1:
        # 取第一个样本
        waveform = waveform[0]

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    time_axis = np.arange(len(waveform)) / sample_rate
    plt.plot(time_axis, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.tight_layout()


import numpy as np
import torch
import torchaudio
from scipy import signal
import numpy as np
import torch
import torchaudio
from scipy import signal


import torch
import numpy as np
import scipy.signal as signal
import torchaudio

def intelligent_slice_and_align(audio_raw, acc_raw):
    """
    将已切片的 Raw Waveforms 转换为 64x64 Log-Spectrograms
    输入: Numpy Arrays or Tensors (Length ~10448)
    输出: Tensor [1, 64, 64] (Log-scale, 未归一化)
    """
    # === 1. 基础配置 (Hardcoded for Determinism) ===
    SR = 10448
    TARGET_LEN = 10448  # 1.0s
    TARGET_H = 64       # Freq bins / Mels
    TARGET_W = 64       # Time frames
    
    N_FFT = 1024        # ~98ms window
    HOP_LENGTH = 149    # (10448 - 1024) // 63 = 149
    
    # === 2. 数据格式化 ===
    if isinstance(audio_raw, torch.Tensor): audio_raw = audio_raw.numpy()
    if isinstance(acc_raw, torch.Tensor): acc_raw = acc_raw.numpy()
    
    # 展平为 1D
    audio_raw = audio_raw.flatten()
    acc_raw = acc_raw.flatten()
    
    # 强制长度对齐 (Padding / Truncate)
    def fix_length(x, length):
        if len(x) < length:
            return np.pad(x, (0, length - len(x)), 'constant')
        elif len(x) > length:
            return x[:length]
        return x

    audio_raw = fix_length(audio_raw, TARGET_LEN)
    acc_raw = fix_length(acc_raw, TARGET_LEN)

    # ==========================================
    # 3. 特征提取 (Audio -> Mel -> Log)
    # ==========================================
    audio_tensor = torch.from_numpy(audio_raw).float().unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, 
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=TARGET_H,
        center=False,      # 严格匹配物理窗口
        power=2.0
    )
    
    audio_mel = mel_transform(audio_tensor) # [1, 64, 64]
    audio_mel = torch.log1p(audio_mel)      # Log-compression

    # ==========================================
    # 4. 特征提取 (Acc -> STFT -> Log)
    # ==========================================

    # B. STFT
    # noverlap = 1024 - 149 = 875
    f, t, Zxx = signal.stft(
        acc_raw, 
        fs=SR, 
        nperseg=N_FFT, 
        noverlap=N_FFT - HOP_LENGTH,
        boundary=None, 
        padded=False
    )
    mag_acc = np.abs(Zxx)
    # mag_acc = np.log1p(mag_acc) # Log-compression
    
    # C. 裁剪与对齐
    # Freq Axis: 取前 64
    if mag_acc.shape[0] >= TARGET_H:
        mag_acc = mag_acc[:TARGET_H, :]
    else:
        pad_h = TARGET_H - mag_acc.shape[0]
        mag_acc = np.pad(mag_acc, ((0, pad_h), (0, 0)), 'constant')
        
    # Time Axis: 强制 64
    if mag_acc.shape[1] > TARGET_W:
        mag_acc = mag_acc[:, :TARGET_W]
    elif mag_acc.shape[1] < TARGET_W:
        pad_w = TARGET_W - mag_acc.shape[1]
        mag_acc = np.pad(mag_acc, ((0, 0), (0, pad_w)), 'edge')
        
    acc_tensor = torch.from_numpy(mag_acc).float().unsqueeze(0) # [1, 64, 64]
    
    return audio_mel, acc_tensor

def intelligent_slice_and_align0(audio_raw, acc_raw, sr=10448, duration=1.0, target_height=64, target_width=64):
    """
    配置:
    - Duration: 1.0s (10448 samples)
    - Input Size: 64x64
    - FFT Window: 1024 (Acc & Audio 保持物理窗口一致)
    """
    
    # 1. 数据准备
    if isinstance(audio_raw, torch.Tensor): audio_raw = audio_raw.numpy()
    if isinstance(acc_raw, torch.Tensor): acc_raw = acc_raw.numpy()
    audio_raw = audio_raw.squeeze(); acc_raw = acc_raw.squeeze()
    
    target_samples = int(duration * sr)
    total_samples = len(audio_raw)
    
    # 2. 能量切片 (寻找最强 1s)
    if total_samples < target_samples:
        pad_len = target_samples - total_samples
        audio_raw = np.pad(audio_raw, (0, pad_len), 'constant')
        acc_raw = np.pad(acc_raw, (0, pad_len), 'constant')
        start_idx = 0
    else:
        energy = audio_raw ** 2
        window = np.ones(target_samples)
        sliding_energy = np.convolve(energy, window, mode='valid')
        start_idx = np.argmax(sliding_energy)
    
    end_idx = start_idx + target_samples
    audio_slice = audio_raw[start_idx:end_idx]
    acc_slice = acc_raw[start_idx:end_idx]
    
    # ==========================================
    # 3. 特征提取 (双模态参数同步)
    # ==========================================
    
    # 🔴 核心决策：两者都用 1024，保证物理窗口时长一致 (98ms)
    n_fft = 1024
    
    # 动态计算 Hop Length 以填满 64 帧
    # Hop = (10448 - 1024) / 63 = 149.58 -> 149 samples (约 14.3ms)
    hop_length = int((target_samples - n_fft) / (target_width - 1))
    
    # --- A. Audio Mel Spectrogram (64 Mels) ---
    audio_tensor = torch.from_numpy(audio_slice).float().unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, 
        n_fft=n_fft,       # 1024
        hop_length=hop_length, 
        n_mels=target_height, # 64 bins
        center=False,      # 关闭 Padding 以严格对齐
        power=2.0
    )
    audio_mel = mel_transform(audio_tensor) # [1, 64, 64]
    # 注意：不在这里做 log，交给外部 normalize_mel_spectrogram

    # --- B. Acc Lowpass STFT (1024 FFT) ---
    # 1. 低通滤波 500Hz
    b, a = signal.butter(4, 500, fs=sr, btype='low') 
    acc_filtered = signal.filtfilt(b, a, acc_slice)
    
    # 2. STFT
    nperseg = n_fft # 1024
    noverlap = nperseg - hop_length # 1024 - 149 = 875
    
    f, t, Zxx = signal.stft(
        acc_filtered, fs=sr, nperseg=nperseg, noverlap=noverlap, 
        boundary=None, padded=False
    )
    mag_acc = np.abs(Zxx)
    
    # 3. 频率轴裁剪 (取前64个线性频点)
    # n_fft=1024 产生 513 个频点。我们取前 64 个 (0~652Hz)
    if mag_acc.shape[0] >= target_height:
        mag_acc = mag_acc[:target_height, :] 
    else:
        pad_h = target_height - mag_acc.shape[0]
        mag_acc = np.pad(mag_acc, ((0, pad_h), (0, 0)), 'constant')

    # 4. 时间轴对齐
    if mag_acc.shape[1] > target_width:
        mag_acc = mag_acc[:, :target_width]
    elif mag_acc.shape[1] < target_width:
        pad_w = target_width - mag_acc.shape[1]
        mag_acc = np.pad(mag_acc, ((0,0), (0, pad_w)), 'edge')
        
    acc_tensor = torch.from_numpy(mag_acc).float().unsqueeze(0)
    
    return audio_mel, acc_tensor

def intelligent_slice_and_align1(audio_raw, acc_raw, sr=10448, duration=1.2, target_height=128, target_width=128):
    """
    功能：
    1. 基于Audio能量，定位并截取 1.2s (默认) 的最强发声片段。
    2. 输出 128x128 的特征图。
    
    输入:
        audio_raw: 原始音频数据
        acc_raw: 原始加速度数据
        sr: 采样率 (default 10448)
        duration: 截取时长 (建议 1.2s 以适配 128 宽度的同时保持原有时间分辨率)
        target_height: 频域维度 (128)
        target_width:  时域维度 (128)
    
    输出:
        audio_mel: Tensor [1, 128, 128]
        acc_mag:   Tensor [1, 128, 128]
    """
    
    # 1. 数据标准化与形状调整
    if isinstance(audio_raw, torch.Tensor): audio_raw = audio_raw.numpy()
    if isinstance(acc_raw, torch.Tensor): acc_raw = acc_raw.numpy()
    
    audio_raw = audio_raw.squeeze()
    acc_raw = acc_raw.squeeze()
    
    # 计算目标采样点数: 1.2 * 10448 = 12537
    target_samples = int(duration * sr)
    total_samples = len(audio_raw)
    
    if total_samples < target_samples:
        # Padding
        pad_len = target_samples - total_samples
        audio_raw = np.pad(audio_raw, (0, pad_len), 'constant')
        acc_raw = np.pad(acc_raw, (0, pad_len), 'constant')
        start_idx = 0
    else:
        # Energy-based Slicing
        energy = audio_raw ** 2
        window = np.ones(target_samples)
        sliding_energy = np.convolve(energy, window, mode='valid')
        start_idx = np.argmax(sliding_energy)
    
    end_idx = start_idx + target_samples
    
    audio_slice = audio_raw[start_idx:end_idx]
    acc_slice = acc_raw[start_idx:end_idx]
    
    # 1. 去除 Audio 的直流偏置 (虽通常很小，但为了保险)
    audio_slice = audio_slice - np.mean(audio_slice)
    
    # 2. 去除 Acc 的重力分量和静态偏置 (至关重要！)
    # 这会将信号中心拉回 0，消除 0Hz 的巨大能量峰值
    acc_slice = acc_slice - np.mean(acc_slice)
    
    # ==========================================
    # 3. 特征提取 (强制对齐 128x128)
    # ==========================================
    
    # 参数计算: 
    n_fft = 512
    # 动态计算 hop_length 以填满 target_width (128)
    # hop = (samples - n_fft) / (frames - 1)
    hop_length = int((target_samples - n_fft) / (target_width - 1))
    
    # --- A. Audio Mel Spectrogram ---
    audio_tensor = torch.from_numpy(audio_slice).float().unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=target_height, # 关键：这里改为 128
        center=False,
        power=2.0
    )
    
    audio_mel = mel_transform(audio_tensor) # [1, 128, 128]

    # --- B. Acc Lowpass STFT ---
    b, a = signal.butter(4, 500, fs=sr, btype='low') # 保持低通滤波不变
    acc_filtered = signal.filtfilt(b, a, acc_slice)
    
    nperseg = 512
    noverlap = nperseg - hop_length
    
    f, t, Zxx = signal.stft(
        acc_filtered, 
        fs=sr, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        boundary=None, 
        padded=False
    )
    
    mag_acc = np.abs(Zxx)
    
    # 3. 裁剪/填充频率轴 
    # n_fft=512 会产生 257 个频点。我们需要前 128 个。
    # 128个频点对应频率范围: 0 ~ 2612 Hz，这完全覆盖了骨传导的有效范围(通常 < 2000Hz)
    if mag_acc.shape[0] >= target_height:
        mag_acc = mag_acc[:target_height, :] 
    else:
        # 如果频点不够（极少情况），进行padding
        pad_h = target_height - mag_acc.shape[0]
        mag_acc = np.pad(mag_acc, ((0, pad_h), (0, 0)), 'constant')

    # 4. 强制对齐时间轴 (防止舍入误差导致 127 或 129)
    if mag_acc.shape[1] > target_width:
        mag_acc = mag_acc[:, :target_width]
    elif mag_acc.shape[1] < target_width:
        pad_w = target_width - mag_acc.shape[1]
        mag_acc = np.pad(mag_acc, ((0,0), (0, pad_w)), 'edge')
        
    acc_tensor = torch.from_numpy(mag_acc).float().unsqueeze(0) # [1, 128, 128]
    
    return audio_mel, acc_tensor


# 把nerseg改为了512取了相位谱的分支
def lowpass_stft_batch_phase(signals, fs=10448, cutoff=1290, filter_order=4, nperseg=256):
    """
    对 batch 信号进行 STFT 变换，并仅返回 0-2000 Hz 的幅度谱和相位谱。
    """

    batch_size, signal_length = signals.shape
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=filter_order, Wn=normal_cutoff, btype='low', analog=False)

    magnitude_list = []
    phase_list = []

    filtered_signal = filtfilt(b, a, signals[0])
    f, t, Zxx = stft(filtered_signal, fs=fs, nperseg=nperseg)

    freq_mask = (f <= cutoff)
    f_sub = f[freq_mask]

    magnitude_sub = np.abs(Zxx[freq_mask, :])
    phase_sub = np.angle(Zxx[freq_mask, :])

    magnitude_list.append(magnitude_sub)
    phase_list.append(phase_sub)

    for i in range(1, batch_size):
        filtered_signal = filtfilt(b, a, signals[i])
        _, _, Zxx = stft(filtered_signal, fs=fs, nperseg=nperseg)
        magnitude_list.append(np.abs(Zxx[freq_mask, :]))
        phase_list.append(np.angle(Zxx[freq_mask, :]))

    magnitude_batch = np.stack(magnitude_list, axis=0)
    phase_batch = np.stack(phase_list, axis=0)

    return f_sub, t, magnitude_batch, phase_batch



def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """计算WGAN-GP的梯度惩罚"""
    # 随机选择一个权重alpha
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    
    # 创建真实样本和虚假样本之间的插值
    # .detach() 是为了不让梯度传播到生成器
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.detach())).requires_grad_(True)
    
    # 计算判别器对插值样本的评分
    d_interpolates = discriminator(interpolates)
    
    # 创建一个与判别器输出形状相同的张量，用于计算梯度
    fake = torch.ones(d_interpolates.size()).to(device)
    
    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True, # 保留计算图，允许二阶导数
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 将梯度展平
    gradients = gradients.view(gradients.size(0), -1)
    
    # 计算梯度的L2范数，并计算惩罚项
    # (||gradient|| - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty



