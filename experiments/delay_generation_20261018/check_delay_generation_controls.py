"""Full-record non-circular offset control; fixed checkpoints, no refitting."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from train_delay_generation import Model
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root;torch.set_num_threads(2)
with np.load(root/'data/data_train.npz') as d:mean=torch.from_numpy(d['y'].mean((0,2)))
with np.load(root/'data/data_val.npz') as d:y=d['y'].copy();ids=d['ids'].copy()
with np.load(root/'data/validation_offset_400ms.npz') as d:x=torch.from_numpy(d['x']).cuda();assert np.array_equal(ids,d['ids'])
results=[]
for r in json.loads((root/'results/results.json').read_text()):
 name=r['route']+'_'+str(r['seed']);m=Model(r['route'],mean).cuda();m.load_state_dict(torch.load(root/'results'/(name+'_weights.pt'),map_location='cuda',weights_only=True));m.eval()
 with torch.no_grad():shift=m(x)[0].cpu().numpy()
 with np.load(root/'results'/(name+'_predictions.npz')) as d:original=d['audio']
 rows=[]
 for g in np.unique(ids):
  ii=ids==g;rows.append(dict(recording=str(g),original_audio_mae=float(np.mean(abs(original[ii]-y[ii]))),noncircular_plus400ms_audio_mae=float(np.mean(abs(shift[ii]-y[ii]))),noncircular_plus400ms_minus_matched_mae=float(np.mean(abs(shift[ii]-y[ii]))-np.mean(abs(original[ii]-y[ii])))))
 results.append(dict(route=r['route'],seed=r['seed'],per_record=rows,macro={k:float(np.mean([z[k] for z in rows])) for k in rows[0] if k!='recording'}))
(root/'results/noncircular_controls.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print([(r['seed'],r['route'],r['macro']) for r in results],flush=True)
