import numpy as np
hop_ms=5
def vad_detect(audio_1d, fs, frame_len, hop_len):
    n = len(audio_1d)
    n_frames = (n - frame_len) // hop_len + 1
    if n_frames <= 0:
        return (0, n)
    rms = np.zeros(n_frames)
    for f in range(n_frames):
        start = f * hop_len
        rms[f] = np.sqrt(np.mean(audio_1d[start:start + frame_len] ** 2))
    top10 = np.percentile(rms, 90)
    threshold = 0.15 * np.mean(rms[rms >= top10])
    vad_frames = rms > threshold
    min_frames = int(30 / hop_ms)
    i = 0
    while i < len(vad_frames):
        if vad_frames[i]:
            j = i
            while j < len(vad_frames) and vad_frames[j]:
                j += 1
            if j - i < min_frames:
                vad_frames[i:j] = False
            i = j
        else:
            i += 1
    merge_frames = int(50 / hop_ms)
    i = 0
    while i < len(vad_frames):
        if not vad_frames[i]:
            j = i
            while j < len(vad_frames) and (not vad_frames[j]):
                j += 1
            if j - i < merge_frames and i > 0 and (j < len(vad_frames)):
                vad_frames[i:j] = True
            i = j
        else:
            i += 1
    voiced = np.where(vad_frames)[0]
    if len(voiced) == 0:
        return (0, n)
    start_sample = voiced[0] * hop_len
    end_sample = min(voiced[-1] * hop_len + frame_len, n)
    return (start_sample, end_sample)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_min, mel_max = (hz_to_mel(fmin), hz_to_mel(fmax))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m]):
            if bins[m] > bins[m - 1]:
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(bins[m], bins[m + 1]):
            if bins[m + 1] > bins[m]:
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank