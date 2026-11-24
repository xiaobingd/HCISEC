import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('outputs/vr261119_10x40_raw_acc');out=root/'conditional_correspondence/raw_common_dynamics'
d=np.load(out/'local_power_private.npz');manifest=json.loads((root/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
groups=list(dict.fromkeys(manifest[int(i)]['record_group'] for i in d['indices']));rates=[]
for stem in groups:
    p=Path(stem);v=pd.read_csv(p.with_name(p.name+'_full_imu.csv'),usecols=['imu_sample_time'])['imu_sample_time'].drop_duplicates().sort_values().to_numpy()
    rates.append(float(1/np.median(np.diff(v))))
result=dict(recording_groups=len(rates),native_rate_hz_min=float(min(rates)),native_rate_hz_median=float(np.median(rates)),native_rate_hz_max=float(max(rates)),native_nyquist_min_hz=float(min(rates)/2),analysis_upper_frequency_hz=450)
(out/'native_rate_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
