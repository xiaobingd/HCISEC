"""Read-only paired-noise controls for complete-cycle flow pilot."""
import os,sys,argparse,json,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch.nn import functional as F
from train_mei_flow_cycle import FlowG,FullG,ReturnF,legacy_predictions,metrics,load,detail_metrics
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output
va=load(args.prior/'full_context_data/full_validation.npz');tr=load(args.prior/'full_context_data/full_train.npz');legacy=load(args.source/'data/mei_val.npz');proto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=proto['normalization_audio'];threshold=proto['audio_energy_threshold']
runs=json.loads((root/'results.json').read_text());assert len(runs) in (2,6)
mean_x=tr['x'].mean(0);mean_y=tr['y'].mean(0);wrong=dict(va);wrong['x']=np.roll(va['x'],1,axis=0);constant=dict(va);constant['x']=np.broadcast_to(mean_x,va['x'].shape).copy();controls=[]
for run in runs:
    seed,route=run['seed'],run['route'];cp=torch.load(root/(route+'_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True);g=FlowG(FullG(),route).cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();rows=[];preds_by_draw=[]
    for draw in range(1 if route=='direct' else 3):
        g.noise_generator.manual_seed(12000+draw);d,preds,full=legacy_predictions(g,f,va,legacy);score=metrics(d,preds,alo,ahi,threshold)
        g.noise_generator.manual_seed(12000+draw);_,wp,_=legacy_predictions(g,f,wrong,legacy)
        g.noise_generator.manual_seed(12000+draw);_,npred,_=legacy_predictions(g,f,constant,legacy)
        wrong_score=metrics(d,wp,alo,ahi,threshold);none_score=metrics(d,npred,alo,ahi,threshold)
        q=[];bl=[]
        with torch.no_grad():
            for i in range(0,len(va['ids']),12):
                aa=torch.as_tensor(full['generated_audio'][i:i+12],device='cuda');qq=torch.round((aa+1)*127.5)/127.5-1;bb=F.avg_pool2d(aa[:,None],3,stride=1,padding=1,count_include_pad=False)[:,0]
                q.append(f(qq).cpu().numpy());bl.append(f(bb).cpu().numpy())
        q,bl=np.concatenate(q),np.concatenate(bl);lookup={str(n):i for i,n in enumerate(full['ids'])};qp=[];bp=[];template=[]
        for n in np.unique(d['ids']):
            idx=lookup[str(n)]
            for j in range(3):
                sl=slice(j*40+8,j*40+73);qp.append(q[idx,:,sl]);bp.append(bl[idx,:,sl]);template.append(mean_y[:,sl])
        cycle=float(abs(preds['cycle_ACC']-d['x']).mean());preds_by_draw.append(preds['generated_audio'])
        rows.append(dict(draw=draw,wrong_record_macro=wrong_score['macro'],constant_ACC_macro=none_score['macro'],wrong_record_audio_mae_change=wrong_score['macro']['audio_mae']-score['macro']['audio_mae'],constant_ACC_audio_mae_change=none_score['macro']['audio_mae']-score['macro']['audio_mae'],quant8_cycle_mae_increase=float(abs(np.array(qp)-d['x']).mean()-cycle),blur3_cycle_mae_increase=float(abs(np.array(bp)-d['x']).mean()-cycle),TRAIN_only_time_template_audio_mae=float(abs(np.array(template)-d['y']).mean())))
    if route!='direct':
        old_sigma=g.sigma;g.sigma=0;g.noise_generator.manual_seed(12000);d,zp,zfull=legacy_predictions(g,f,va,legacy);zero=dict(metrics=metrics(d,zp,alo,ahi,threshold),sharp=detail_metrics(zp['generated_audio'],d['y'],threshold,alo,ahi));g.sigma=old_sigma
        g.steps=1;g.noise_generator.manual_seed(12000);d,one,_=legacy_predictions(g,f,va,legacy);one_step=dict(metrics=metrics(d,one,alo,ahi,threshold),sharp=detail_metrics(one['generated_audio'],d['y'],threshold,alo,ahi));g.steps=5
        # Native full axes as a private first4source diagnostic, fixed same sources.
        pp=load(root/(route+'_'+str(seed)+'_plots.npz'));ii=[np.flatnonzero(zfull['ids']==n)[0] for n in pp['ids']]
        np.savez_compressed(root/(route+'_'+str(seed)+'_zero_noise_plots.npz'),generated_audio=zfull['generated_audio'][ii],ids=pp['ids'])
        diversity=float(np.mean(np.std(np.stack(preds_by_draw),axis=0)))
    else:zero=None;one_step=None;diversity=0.
    r=dict(seed=seed,route=route,draws=rows,zero_prior_noise=zero,one_NFE_same_weights=one_step,mean_pointwise_std_across3draws=diversity);controls.append(r);(root/'controls.json').write_text(json.dumps(dict(runs=controls,scope='Same Gaussian noise for real/wrong/constantACC comparisons. No best-of-N, no draw-average Mel reconstruction. ConstantACC uses TRAIN-only mean; template TRAIN-only mean audio. Zero sigma and1NFE diagnostic only. Quant8/blur3 act on normalizedMel not waveform. No identity proof.'),indent=2));print('CONTROL',seed,route,json.dumps(dict(diversity=diversity,wrong_mae_change=float(np.mean([x['wrong_record_audio_mae_change'] for x in rows])))),flush=True);del g,f,cp;torch.cuda.empty_cache()
with zipfile.ZipFile(root/'review.zip','a',zipfile.ZIP_DEFLATED) as z:
    z.write(root/'controls.json','controls.json')
    for path in root.glob('*_zero_noise_plots.npz'):z.write(path,path.name)
print('AUDIT_DONE',flush=True)
