"""ACC inference interventions, not model selection or identity verification."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_activity_alignment as lib
import numpy as np
import torch
ROOT=Path('outputs/activity_alignment_20261018')
torch.set_num_threads(2)
tr=lib.load(ROOT,'train');va=lib.load(ROOT,'val');groups=np.unique(va['ids']);permutation=np.arange(len(va['ids']))
for k,g in enumerate(groups):
 ii=np.flatnonzero(va['ids']==g);jj=np.flatnonzero(va['ids']==groups[(k+1)%len(groups)]);permutation[ii]=np.resize(jj,len(ii))
out=[]
for result in json.loads((ROOT/'results.json').read_text()):
 route=result['route'];model=lib.Model(route,lib.tensor(tr['y'].mean((0,2))))
 model.load_state_dict(torch.load(ROOT/(route+'_weights.pt'),weights_only=True,map_location='cpu'));model.eval()
 with torch.no_grad():
  p,a,*_=model(lib.tensor(va['x']));shift,sa,*_=model(torch.roll(lib.tensor(va['x']),40,-1));wrong,wa,*_=model(lib.tensor(va['x'][permutation]))
 p,shift,wrong=p.numpy(),shift.numpy(),wrong.numpy();a,sa,wa=a.sigmoid().numpy(),sa.sigmoid().numpy(),wa.sigmoid().numpy()
 rows=[]
 for g in groups:
  ix=va['ids']==g;truth=va['y'][ix];err=np.mean(abs(p[ix]-truth))
  rows.append(dict(recording=str(g),matched_audio_mae=float(err),shift400ms_minus_matched_mae=float(np.mean(abs(shift[ix]-truth))-err),wrong_record_ACC_minus_matched_mae=float(np.mean(abs(wrong[ix]-truth))-err),matched_activity_auc=lib.auc(va['activity'][ix],a[ix]),shift400ms_activity_auc=lib.auc(va['activity'][ix],sa[ix]),wrong_record_ACC_activity_auc=lib.auc(va['activity'][ix],wa[ix])))
 out.append(dict(route=route,per_record=rows,macro={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='recording'}))
(ROOT/'conditioning_controls.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
