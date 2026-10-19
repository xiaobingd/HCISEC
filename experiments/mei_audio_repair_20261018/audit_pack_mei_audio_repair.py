"""Read-only controls and compact private review; excludes source TEST/auth files."""
import os,sys,argparse,json,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from train_mei_audio_repair import extra_metrics,ReturnF,load

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root;res=root/'results';out=root/'review';out.mkdir(exist_ok=True);torch.set_num_threads(2)
runs=json.loads((res/'results.json').read_text());assert len(runs)==8
conservative=json.loads((root/'conservative_results/results.json').read_text());assert len(conservative)==4
runs+=conservative
protocol=json.loads((res/'protocol.json').read_text());alo,ahi=protocol['normalization_audio'];threshold=protocol['audio_energy_threshold']
val=load(args.source/'data/mei_val.npz');train=load(args.source/'data/mei180_train.npz')
ug=np.unique(val['ids']);names=[g for g in ug if str(g).startswith('0-9__')][:2]+[g for g in ug if str(g).startswith('word__')][:2]
indices=[np.flatnonzero(val['ids']==g)[0] for g in names]
np.savez_compressed(out/'references.npz',**{k:val[k][indices] for k in ['x','y','ids']})
template=train['y'].reshape(180,4,80,81).mean(0);pred=np.tile(template,(60,1,1))
time_controls=extra_metrics(dict(generated_audio=pred),val,alo,ahi,threshold)
np.savez_compressed(out/'time_template_predictions.npz',generated_audio=pred[indices])
allcontrols=[];initial=[]
source_runs=json.loads((args.source/'results_long/results.json').read_text())
for seed in [42,43]:
    d=load(args.source/'results_long'/('mei180_long_'+str(seed)+'_predictions.npz'))
    old=next(r for r in source_runs if r['seed']==seed)
    initial.append(dict(seed=seed,validation=old['validation'],extra_validation=extra_metrics(d,val,alo,ahi,threshold)))
    np.savez_compressed(out/('initial_'+str(seed)+'_predictions.npz'),**{k:v[indices] for k,v in d.items()})
    routes=[('initial',args.source/'results_long'/('mei180_long_'+str(seed)+'_weights.pt'),d)]
    for r in [r for r in runs if r['seed']==seed]:
        name=r['route']+'_'+str(seed);folder=root/'conservative_results' if r['route'] in ['energy_detail','activity_detail'] else res;d=load(folder/(name+'_predictions.npz'))
        np.savez_compressed(out/(name+'_predictions.npz'),**{k:v[indices] for k,v in d.items()})
        routes.append((r['route'],folder/(name+'_weights.pt'),d))
    for name,cp_path,d in routes:
        f=ReturnF().cuda();cp=torch.load(cp_path,map_location='cuda',weights_only=True);f.load_state_dict(cp['f']);f.eval();outs={k:[] for k in ['clean','quant8','real','real_quant8','wrong']}
        ids=val['ids'];perm=np.empty(len(ids),dtype=int)
        for k,g in enumerate(ug):perm[np.flatnonzero(ids==g)]=np.flatnonzero(ids==ug[(k+1)%len(ug)])
        with torch.no_grad():
            for start in range(0,len(ids),24):
                stop=start+24;ga=torch.as_tensor(d['generated_audio'][start:stop],device='cuda');real=torch.as_tensor(val['y'][start:stop],device='cuda');wrong=torch.as_tensor(d['generated_audio'][perm[start:stop]],device='cuda')
                q=lambda y:((y.clamp(-1,1)+1)*127.5).round()/127.5-1
                for key,z in [('clean',f(ga)),('quant8',f(q(ga))),('real',f(real)),('real_quant8',f(q(real))),('wrong',f(wrong))]:outs[key].append(z.cpu().numpy())
        error={k:np.mean(abs(np.concatenate(v)-val['x']),axis=(1,2)) for k,v in outs.items()}
        macro={k:float(np.mean([v[ids==g].mean() for g in ug])) for k,v in error.items()}
        power=lambda a:np.log10(np.mean(10**((a-1)*(ahi-alo)/20),axis=1).clip(1e-12))
        old_audio=load(args.source/'results_long'/('mei180_long_'+str(seed)+'_predictions.npz'))['generated_audio']
        actual_delta=abs(power(d['generated_audio'])-power(old_audio))
        allcontrols.append(dict(route=name,seed=seed,clean_cycle_mae=macro['clean'],quant8_cycle_mae_increase=macro['quant8']-macro['clean'],quant8_REALaudio_ACC_mae_increase=macro['real_quant8']-macro['real'],wrong_generated_audio_cycle_mae_increase=macro['wrong']-macro['clean'],mean_absolute_logMelpower_change_from_initial=float(actual_delta.mean()),output_clip_fraction=float(np.mean(abs(d['generated_audio'])>=.999999))))
        del f,cp;torch.cuda.empty_cache()
controls=dict(initial=initial,time_template_extra=time_controls,time_template_audio_mae=float(np.mean([abs(pred[val['ids']==g]-val['y'][val['ids']==g]).mean() for g in ug])),perturbation=allcontrols,scope='Quantization is normalizedMelnumeric perturbation, not an audible channel test; input dependency is not identity authentication. AllVAL previously explored, no test/auth read.')
(out/'controls.json').write_text(json.dumps(controls,indent=2))
(out/'protocol.json').write_bytes((res/'protocol.json').read_bytes())
(out/'results.json').write_text(json.dumps(runs,indent=2))
(out/'conservative_protocol.json').write_bytes((root/'conservative_results/protocol.json').read_bytes())
for f in res.glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
for f in (root/'conservative_results').glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in out.iterdir():z.write(f,'review/'+f.name)
print('BUNDLE_BYTES',(root/'review.zip').stat().st_size)
