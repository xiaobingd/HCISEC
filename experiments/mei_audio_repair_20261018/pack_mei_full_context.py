"""Common-target controls for long-context adaptation; separate from old full-window scores."""
import os,argparse,json,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from train_mei_full_context import metrics,FullG,ReturnF
from train_mei_audio_repair import load,corr,detail,extra_metrics
import torch
torch.set_num_threads(2)

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--study',type=Path,required=True);args=p.parse_args();root=args.study;res=root/'full_context_results';out=root/'full_context_review';out.mkdir(exist_ok=True)
runs=json.loads((res/'results.json').read_text());assert len(runs)==2
proto=json.loads((root/'results/protocol.json').read_text());alo,ahi=proto['normalization_audio'];threshold=proto['audio_energy_threshold']
train=load(root/'full_context_data/full_train.npz');full_val=load(root/'full_context_data/full_validation.npz');legacy=load(args.source/'data/mei_val.npz');mean=train['y'].mean(0);blocks=np.stack([mean[:,j*40+8:j*40+73] for j in range(3)]);template=np.tile(blocks,(60,1,1))
controls=[]
for run in runs:
    seed=run['seed'];common=load(res/('full_'+str(seed)+'_common_predictions.npz'));full=load(res/('full_'+str(seed)+'_predictions.npz'))
    d=dict(x=common['real_ACC'],y=common['real_audio'],ids=common['ids']);ug=np.unique(d['ids']);lookup={str(n):i for i,n in enumerate(full['ids'])};wrong=[]
    for k,n in enumerate(ug):
        idx=lookup[str(ug[(k+1)%len(ug)])]
        wrong.extend([full['generated_audio'][idx,:,j*40+8:j*40+73] for j in range(3)])
    wrong=np.array(wrong);records=[]
    for name in ug:
        ii=d['ids']==name
        records.append(dict(recording=str(name),time_template_audio_mae=float(abs(template[ii]-d['y'][ii]).mean()),time_template_detail_corr=corr(detail(template[ii]),detail(d['y'][ii])),wrong_record_audio_mae_increase=float(abs(wrong[ii]-d['y'][ii]).mean()-abs(common['generated_audio'][ii]-d['y'][ii]).mean())))
    controls.append(dict(seed=seed,macro={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='recording'},per_record=records))
    # Deterministic first2sources/task, middle of first originalwindow8..72frames.
    names=[n for n in ug if str(n).startswith('0-9__')][:2]+[n for n in ug if str(n).startswith('word__')][:2];ii=[np.flatnonzero(d['ids']==n)[0] for n in names]
    old=load(args.source/'results_long'/('mei180_long_'+str(seed)+'_predictions.npz'));oldlookup={str(n):np.flatnonzero(legacy['ids']==n)[0] for n in ug}
    np.savez_compressed(out/('full_'+str(seed)+'_plots.npz'),real_audio=d['y'][ii],real_ACC=d['x'][ii],generated_audio=common['generated_audio'][ii],cycle_ACC=common['cycle_ACC'][ii],old_audio=np.array([old['generated_audio'][oldlookup[str(n)],:,8:-8] for n in names]),old_cycle=np.array([old['cycle_ACC'][oldlookup[str(n)],:,8:-8] for n in names]),time_template=template[ii],ids=d['ids'][ii])
    fi=[lookup[str(n)] for n in names];vi=[np.flatnonzero(full_val['ids']==n)[0] for n in names]
    g=FullG().cuda();f=ReturnF().cuda();cp=torch.load(args.source/'results_long'/('mei180_long_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True);g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval()
    with torch.no_grad():oldfull=g(torch.as_tensor(full_val['x'][vi],device='cuda')).cpu().numpy()
    np.savez_compressed(out/('full_'+str(seed)+'_whole_record_plots.npz'),real_audio=full_val['y'][vi],real_ACC=full_val['x'][vi],initial_full_audio=oldfull,generated_audio=full['generated_audio'][fi],cycle_ACC=full['cycle_ACC'][fi],time_template=np.repeat(mean[None],4,axis=0),ids=full_val['ids'][vi])
    del g,f,cp;torch.cuda.empty_cache()
public=dict(runs=runs,controls=controls,protocol=json.loads((res/'protocol.json').read_text()),scope='Same65interiorframes fromfirst3legacycrops /60records, not olderall240/full81metrics. Fullcontext permits futureACC, noMIC inference and no timestamp edits. No identityclaim.')
(out/'summary.json').write_text(json.dumps(public,indent=2))
for f in res.glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
with zipfile.ZipFile(root/'full_context_review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in out.iterdir():z.write(f,'full_context_review/'+f.name)
print('FULL_BUNDLE',(root/'full_context_review.zip').stat().st_size)
