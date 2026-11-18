"""Read-only detail and bottleneck controls; no checkpoint selection."""
import os,json,argparse,zipfile
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_mei_sharp_cycle import FullG,ReturnF,SharpG,HighpassG,load,legacy_predictions,detail_metrics

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
va=load(args.prior/'full_context_data/full_validation.npz');legacy=load(args.source/'data/mei_val.npz');proto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=proto['normalization_audio'];threshold=proto['audio_energy_threshold']
rows=[]
for seed in [42,43]:
    routes=json.loads((args.output/'protocol.json').read_text())['routes']
    for route in ['initial']+routes:
        path=args.prior/'full_context_results'/('full_'+str(seed)+'_weights.pt') if route=='initial' else args.output/(route+'_'+str(seed)+'_weights.pt')
        cp=torch.load(path,map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda()
        if route.endswith('_residual'):g=SharpG(g).cuda()
        if route=='highpass':g=HighpassG(g).cuda()
        g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval()
        d,preds,full=legacy_predictions(g,f,va,legacy);sharp=detail_metrics(preds['generated_audio'],d['y'],threshold,alo,ahi)
        # Return perturbations on full recording before scoring common interiors.
        q=[];bl=[]
        with torch.no_grad():
            for i in range(0,len(va['ids']),12):
                aa=torch.as_tensor(full['generated_audio'][i:i+12],device='cuda')
                qq=torch.round((aa+1)*127.5)/127.5-1
                bb=F.avg_pool2d(aa[:,None],3,stride=1,padding=1,count_include_pad=False)[:,0]
                q.append(f(qq).cpu().numpy());bl.append(f(bb).cpu().numpy())
        lookup={str(n):i for i,n in enumerate(full['ids'])};q=np.concatenate(q);bl=np.concatenate(bl);qp=[];bp=[];wrong=[]
        names=np.unique(d['ids'])
        for k,n in enumerate(names):
            idx=lookup[str(n)];wi=lookup[str(names[(k+1)%len(names)])]
            for j in range(3):
                sl=slice(j*40+8,j*40+73);qp.append(q[idx,:,sl]);bp.append(bl[idx,:,sl]);wrong.append(full['generated_audio'][wi,:,sl])
        cycle=abs(preds['cycle_ACC']-d['x']).mean();row=dict(seed=seed,route=route,sharp=sharp,quant8_cycle_mae_increase=float(abs(np.array(qp)-d['x']).mean()-cycle),blur3_cycle_mae_increase=float(abs(np.array(bp)-d['x']).mean()-cycle),wrong_record_audio_mae_increase=float(abs(np.array(wrong)-d['y']).mean()-abs(preds['generated_audio']-d['y']).mean()))
        rows.append(row);print(json.dumps(row),flush=True);del g,f,cp;torch.cuda.empty_cache()
output=dict(runs=rows,scope='Read-only controls on repeatedly explored VAL. Quant8/3x3 blur act on normalized Mel, not waveform. Wrong-record input dependence and detail metrics do not establish speaker identity or harmonic truth.')
(args.output/'controls.json').write_text(json.dumps(output,indent=2))
with zipfile.ZipFile(args.output/'review.zip','a',zipfile.ZIP_DEFLATED) as z:z.write(args.output/'controls.json','controls.json')
