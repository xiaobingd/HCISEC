"""Existing windows, original per-axis filter, preserve signed axes before norm."""
import json,numpy as np
from pathlib import Path
from scipy import signal
from scipy.signal.windows import tukey
base=Path('outputs/imu_original_frontend_20261016');out=Path('outputs/harmonic_scaffold_20261017');out.mkdir(exist_ok=True)
rows=json.loads((base/'manifest.json').read_text())['samples'];fir=signal.firwin(101,[40,499],pass_zero=False,fs=1000)
cache={}
for split in ['train','val']:
    a=[];ids=[]
    for r in rows:
        if r['split']!=split:continue
        g=r['recording']
        if g not in cache:
            with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as z:x=z['acc_raw']
            x=signal.filtfilt(fir,[1.],x-x.mean(0),axis=0,padlen=min(303,len(x)-1))
            cache[g]=(x*tukey(len(x),alpha=min(202/len(x),1))[:,None]).astype('float32')
        b=int(round(r['start_s']*1000));w=cache[g][b:b+800].T
        assert w.shape==(3,800)
        a.append(w);ids.append(g)
    expected=np.load(base/split/'recording_ids.npy');assert np.array_equal(ids,expected)
    np.savez_compressed(out/('axes_'+split+'.npz'),axes=np.array(a),ids=np.array(ids))
    print(split,len(a))
(out/'protocol.json').write_text(json.dumps(dict(source='same personal raw IMU',split='existing20train4val; no test or28reserve',filter='original FIR101 40-499 Hz per axis, record DC removal and Tukey unchanged',representation='original magnitude frontend vs signed XYZ spectrum/pitch salience; ablation explicitly separate',goal='voicing+F0 then harmonic scaffold, not authentication or full speech reconstruction',shared_rules='no per-user filter/gain, all calibration train only'),indent=2))
