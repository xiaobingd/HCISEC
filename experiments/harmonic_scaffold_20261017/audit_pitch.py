"""Does a smooth pitch track match its input, beyond other recordings/time shifts?"""
from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parent;d=np.load(p/'validation_scaffold.npz');y=d['reference_f0'];v=d['reference_voice'].copy();ids=d['ids']
v[:,:4]=False;v[:,-4:]=False
def error(f,target):return abs(1200*np.log2(np.maximum(f,1)/np.nan_to_num(target,nan=234.51839242421903)))
interior=np.ones_like(v);interior[:,:4]=False;interior[:,-4:]=False
out={'all_voiced_control_f1':float(2*v.sum()/(interior.sum()+v.sum()))}
for name in ('original_pitch','signed_axes_pitch','signed_axes_contrast','constant_train_pitch','physical_acc_pitch'):
    f=d[name+'_f0'];e=error(f,y);matched=float(np.mean(e[v]<50));per={str(g):float(np.mean(e[ids==g][v[ids==g]]<50)) for g in np.unique(ids)}
    wrong=[]
    for i in range(len(f)):
        for j in np.flatnonzero(ids!=ids[i]):wrong.extend((error(f[i],y[j])[v[j]]<50).tolist())
    shifted=float(np.mean(error(np.roll(f,20,axis=1),y)[v]<50))
    out[name]={'matched_within50':matched,'wrong_record_within50':float(np.mean(wrong)),'shift200ms_within50':shifted,'per_record_within50':per}
(p/'pitch_correspondence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
