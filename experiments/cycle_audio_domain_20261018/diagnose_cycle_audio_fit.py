"""TRAIN-vs-reused-VAL fit gap; checkpoints unchanged, no further tuning."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from train_cycle_audio_domain import Generator,load,smooth
from train_cycle_resnet_curriculum import ResNetAudio
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2);out=[]
for kind,folder in [('temporal',args.root/'results'),('resnet',args.root/'resnet_results')]:
 for row in json.loads((folder/'results.json').read_text()):
  route,seed=row['route'],row['seed'];g=(ResNetAudio() if kind=='resnet' else Generator(251,80,True)).cuda();f=Generator(80,251).cuda();cp=torch.load(folder/(route+'_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True);g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();S=smooth if route=='audio_anchor_smooth' else lambda a:a;rr={}
  for split in ['train','val']:
   d=load(args.data/('data_'+split+'.npz'));x=torch.from_numpy(d['x']).cuda();y=torch.from_numpy(d['y']).cuda()
   with torch.no_grad():ga=g(x);rec=f(S(ga));am=abs(ga-y).mean((1,2)).cpu().numpy();cm=abs(rec-x).mean((1,2)).cpu().numpy()
   rr[split]=dict(audio_mae=float(np.mean([am[d['ids']==n].mean() for n in np.unique(d['ids'])])),ACC_cycle_mae=float(np.mean([cm[d['ids']==n].mean() for n in np.unique(d['ids'])])))
  out.append(dict(route=route,seed=seed,fit=rr));print(out[-1],flush=True)
(args.root/'training_fit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
