"""Expand crops inside the existing TRAIN recordings; freeze all frontend scales."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import ast, json, hashlib
from pathlib import Path
import numpy as np
from scipy import signal, ndimage

ROOT = Path('outputs/cycle_generalization_20261018')
OLD = Path('outputs/activity_alignment_20261018')
RAW = Path('outputs/raw_imu_processed_20261014')
ROOT.mkdir(exist_ok=True)
manifest = json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text())
proto = json.loads((OLD/'data_protocol.json').read_text())
norm = manifest['normalization']['audio']
lo, hi = proto['scales']['spec_lo_hi']
mu = np.array(proto['scales']['energy_mean'])[:, None]
sd = np.array(proto['scales']['energy_std'])[:, None]
source = Path(r'E:\Edge下载\train_core\process_imu_data.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'hz_to_mel', 'mel_to_hz', 'create_mel_filterbank'}
env = {'np': np}
exec(compile(ast.Module(body=[n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(source), 'exec'), env)
fb = env['create_mel_filterbank'](16000, 512, 80, 20, 7500)
train_rows = [r for r in manifest['samples'] if r['split'] == 'train']
groups = sorted({r['recording'] for r in train_rows})
assert len(groups) == 20
cache, audit = {}, []
bands = [(2, 10), (10, 40), (40, 100), (100, 200), (200, 450), (450, 490)]

def read_train(g):
    assert g in groups
    with np.load(RAW/(g+'.npz')) as d:
        x = d['acc_raw'].copy(); a = d['audio'].copy(); fs = int(d['audio_fs'])
    x -= x.mean(0)
    if fs != 16000:
        a = signal.resample(a, int(len(a)*16000/fs))
    channels = []
    for xx in [x]+[signal.sosfiltfilt(signal.butter(4, b, btype='bandpass', fs=1000, output='sos'), x, axis=0) for b in bands]:
        for ms in (50, 250):
            power = np.maximum(ndimage.uniform_filter1d(xx*xx, size=ms, axis=0, mode='reflect'), 0)
            channels.extend([np.log10(np.sqrt(power[:, k])+1e-12) for k in range(3)])
            channels.append(np.log10(np.sqrt(power.sum(1))+1e-12))
    ar = np.log10(np.sqrt(np.maximum(ndimage.uniform_filter1d(a*a, size=320, mode='reflect'), 0))+1e-9)
    return x, a, np.array(channels), ar

def crop(g, start):
    x, a, e, ar = cache[g]
    b = round(start*1000); ai = round(start*16000)
    xx = x[b:b+800].T; aa = a[ai:ai+12800]
    assert xx.shape == (3, 800) and aa.shape == (12800,)
    _, _, z = signal.stft(xx, fs=1000, nperseg=128, noverlap=118, nfft=128, axis=-1)
    spec = (2*(20*np.log10(abs(z)+1e-12)-lo)/(hi-lo)-1).reshape(195, 81)
    ix = np.minimum(b+np.arange(81)*10, e.shape[1]-1)
    acc = np.concatenate([spec, (e[:, ix]-mu)/sd]).astype('float32')
    _, _, az = signal.stft(aa, fs=16000, nperseg=512, noverlap=352)
    y = np.clip(2*(10*np.log10(fb@abs(az)**2+1e-10)-norm[0])/(norm[1]-norm[0])-1, -1, 1).astype('float32')
    aix = np.minimum(np.rint((start+np.arange(81)*.01)*16000).astype(int), len(ar)-1)
    frac = float(np.mean(ar[aix] > proto['audio_activity_threshold']))
    return acc, y, frac

def union_seconds(starts):
    end = -1.; total = 0.
    for s in sorted(starts):
        total += max(0., s+.8-max(s, end)); end = max(end, s+.8)
    return total

xs, ys, ids, starts, fractions = [], [], [], [], []
fixed = dict(np.load(OLD/'data_train.npz'))
fixed_fractions = []
max_x_error = max_y_error = 0.
for g in groups:
    cache[g] = read_train(g)
    x, a, _, _ = cache[g]
    duration = min(len(x)/1000, len(a)/16000)
    # A finite crop bank, randomized at training time. No claims of independent new recordings.
    grid = np.arange(0, int(np.floor((duration-.8)*1000))+1, 40)/1000
    original = [r['start_s'] for r in train_rows if r['recording'] == g]
    for t in grid:
        acc, y, frac = crop(g, float(t))
        xs.append(acc); ys.append(y); ids.append(g); starts.append(t); fractions.append(frac)
    audit.append(dict(recording=g, duration=duration, crops=len(grid), old_coverage_seconds=union_seconds(original), new_coverage_seconds=union_seconds(grid)))
    print(g, len(grid), flush=True)
for i, row in enumerate(train_rows):
    x, y, frac = crop(row['recording'], row['start_s'])
    assert fixed['ids'][i] == row['recording']
    max_x_error = max(max_x_error, float(np.max(abs(x-fixed['x'][i]))))
    max_y_error = max(max_y_error, float(np.max(abs(y-fixed['y'][i]))))
    fixed_fractions.append(frac)
assert max_x_error < 1e-5 and max_y_error < 1e-5, (max_x_error, max_y_error)
np.savez_compressed(ROOT/'crop_bank_train.npz', x=np.array(xs), y=np.array(ys), ids=np.array(ids), starts=np.array(starts), activity_fraction=np.array(fractions))
np.savez_compressed(ROOT/'fixed_train.npz', x=fixed['x'], y=fixed['y'], ids=fixed['ids'], activity_fraction=np.array(fixed_fractions))
holdout = sorted(np.random.default_rng(20261018).choice(groups, 4, replace=False).tolist())
protocol = dict(source='one person OpenVR vAccel, existing20TRAIN only for new crops', test_and_auth_reserve='not opened', external_validation='existing4 reused exploratory VAL, unchanged original16 windows', crop_bank=dict(step_ms=40, duration_ms=800, count=len(ids), sampler='8 distinct recordings/update, 75% select crop with audio-active fraction>=0.1 when available;25% uniform. Same rule fixed/dense. TRAIN labels only; no microphone inference', independent_new_recordings=0), coverage=audit, old_union_seconds=sum(r['old_coverage_seconds'] for r in audit), new_union_seconds=sum(r['new_coverage_seconds'] for r in audit), raw_total_seconds=sum(r['duration'] for r in audit), frontend='unchanged251XYZ magnitude+energy and80Mel, exact prior crop functions/parameters; no per-user/axis/window normalization; frozen previous20TRAIN scales, not refit', parity_max_absolute_error=dict(x=max_x_error,y=max_y_error), internal_holdout_recordings=holdout, internal_split='16 fit /4 early-stop. All crops of holdout excluded from gradient updates. Frozen prior calibration includes the20 originalTRAIN; internal split is gradient-held-out, not completely preprocessing-blind. ExternalVAL never used for checkpoint selection.', scale_source_sha256=hashlib.sha256((OLD/'data_protocol.json').read_bytes()).hexdigest(), scope='coverage expansion and sampling of already authorizedTRAIN; overlapping crops are not independent utterances')
(ROOT/'data_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print(json.dumps({k:protocol[k] for k in ['old_union_seconds','new_union_seconds','raw_total_seconds','parity_max_absolute_error','internal_holdout_recordings']}), flush=True)
