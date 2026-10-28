import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """计算WGAN-GP的梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.detach())).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def plot_magnitude_spectrum(mag, title=None):
    """可视化频谱图 (80x80)"""
    mag = mag.squeeze().cpu().detach().numpy()
    plt.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Value")
    if title is not None:
        plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Freq Bin")
    plt.tight_layout()


def plot_waveform(waveform, sample_rate=1000, title='Waveform'):
    """绘制波形图"""
    if len(waveform.shape) > 1:
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
