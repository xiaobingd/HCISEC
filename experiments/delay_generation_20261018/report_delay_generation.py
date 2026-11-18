"""Shared-scale plots and fixed input-free controls, no test/reserve access."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/delay_generation_20261018')
def basis(t):return np.stack([np.ones_like(t)]+[f(2*np.pi*hz*t) for hz in [.25,.5,1,2,4,8] for f in [np.sin,np.cos]],-1)
def main():
 tr=dict(np.load(ROOT/'data_train.npz'));va=dict(np.load(ROOT/'data_val.npz'));results=json.loads((ROOT/'results.json').read_text());ids=va['ids'];groups=np.unique(ids)
 noncircular=json.loads((ROOT/'noncircular_controls.json').read_text())
 for r in results:
  nc=next(c for c in noncircular if c['route']==r['route'] and c['seed']==r['seed']);r['macro'].update({k:v for k,v in nc['macro'].items() if k.startswith('noncircular')})
  for row in r['per_record']:
   match=next(c for c in nc['per_record'] if c['recording']==row['recording']);row.update({k:v for k,v in match.items() if k.startswith('noncircular')})
 rows=json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text())['samples'];starts={s:np.array([r['start_s'] for r in rows if r['split']==s]) for s in ['train','val']}
 train_t=starts['train'][:,None]+np.arange(81)[None]*.01;val_t=starts['val'][:,None]+np.arange(81)[None]*.01;a=basis(train_t).reshape(-1,13);b=basis(val_t).reshape(-1,13);y=tr['y'].transpose(0,2,1).reshape(-1,80)
 weight=np.concatenate([np.full(81,1/np.sum(tr['ids']==g)) for g in tr['ids']]);weight/=weight.mean();w=solve(a.T@(a*weight[:,None])+10*np.eye(13),a.T@(y*weight[:,None]),assume_a='pos');time_pred=(b@w).reshape(-1,81,80).transpose(0,2,1)
 controls=[];mean=tr['y'].mean(0)
 for g in groups:
  ii=ids==g;controls.append(dict(recording=str(g),mean_template_mae=float(np.mean(abs(va['y'][ii]-mean))),time_only_mae=float(np.mean(abs(va['y'][ii]-time_pred[ii])))))
 control_macro={k:float(np.mean([r[k] for r in controls])) for k in controls[0] if k!='recording'}
 summary={}
 for route in ['temporal_only','dual_unconstrained','dual_preserve']:
  rr=[r for r in results if r['route']==route];metrics={}
  for k in rr[0]['macro']:
   vals=[r['macro'][k] for r in rr];metrics[k]=dict(mean=float(np.mean(vals)),std=float(np.std(vals)),values=vals)
  retention={k:dict(mean=float(np.mean([r['acc_retention_probe'][k] for r in rr])),std=float(np.std([r['acc_retention_probe'][k] for r in rr]))) for k in rr[0]['acc_retention_probe']}
  summary[route]=dict(seeds=[r['seed'] for r in rr],metrics=metrics,acc_retention_probe=retention,positive_wrong_record_gap_count=sum(r['macro']['wrong_record_ACC_minus_matched_mae']>0 for r in rr))
 public=dict(scope='same personal20train4REUSEDval;58/16 fixed targets. 3seeds do not add independent recordings. No identitymetric, no finaltest. Newfrontend/context onALL routes.',controls_macro=control_macro,controls_per_record=controls,time_control='TRAIN-only record-balanced ridge10 from fixed elapsed-time sin/cos .25/.5/1/2/4/8Hz; noACC/noVALaudio input',summary=summary,runs=results)
 (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
 routes=list(summary);names=['Temporal only','Dual, no preservation loss','Dual + ACC preservation'];xx=np.arange(3);fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 axs[0].bar(xx,[summary[r]['metrics']['audio_mae']['mean'] for r in routes],yerr=[summary[r]['metrics']['audio_mae']['std'] for r in routes],capsize=4);axs[0].axhline(control_macro['mean_template_mae'],ls='--',c='black',label='Train mean template');axs[0].axhline(control_macro['time_only_mae'],ls=':',c='grey',label='Elapsed-time-only ridge');axs[0].set_title('Audio MAE: mean +/- seed SD');axs[0].legend(fontsize=7)
 for i,k in enumerate(['magnitude','energy','complex']):axs[1].bar(xx+(i-1)*.23,[summary[r]['acc_retention_probe'][k]['mean'] for r in routes],width=.23,label=k)
 axs[1].axhline(1,ls='--',c='black');axs[1].set_title('ACC probe MSE / constant baseline');axs[1].legend(fontsize=7)
 for i,k in enumerate(['noncircular_plus400ms_minus_matched_mae','wrong_record_ACC_minus_matched_mae']):axs[2].bar(xx+(i-.5)*.3,[summary[r]['metrics'][k]['mean'] for r in routes],width=.3,label='Real +400ms offset' if i==0 else 'Wrong recording')
 axs[2].axhline(0,c='black');axs[2].set_title('Intervention MAE - correct input\nPositive: correct input better');axs[2].legend(fontsize=7)
 for ax in axs:ax.set_xticks(xx,names,rotation=12,fontsize=8)
 fig.suptitle('Delay-tolerant cross-modal generation: three seeds, four reused validation recordings; NOT identity verification',fontsize=11);fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
 # Seed42 fixed before training; first frozen window of each record, no selection by appearance.
 preds={r:dict(np.load(ROOT/(r+'_42_predictions.npz'))) for r in routes}
 # Inspection only: these reused VAL embeddings must not train an authentication head.
 raw=np.concatenate([va['x'][:,:,:81].mean(2),va['x'][:,:,:81].std(2)],1)
 for r in routes:
  zz=preds[r]['latent'][:,:,:81];padded=np.zeros((len(ids),80,81),dtype='float32');padded[:,:zz.shape[1]]=zz
  representation=np.concatenate([raw,padded.mean(2),padded.std(2)],1);assert np.array_equal(representation[:,:1282],raw)
  np.savez_compressed(ROOT/(r+'_42_validation_features_inspection_only.npz'),representation=representation,ids=ids)
 fig,axs=plt.subplots(6,4,figsize=(17,15),layout='constrained')
 for c,g in enumerate(groups):
  i=np.flatnonzero(ids==g)[0];acc=va['x'][i,:195].reshape(3,65,129).mean(0);axs[0,c].imshow(acc,origin='lower',aspect='auto',extent=[0,1.28,0,500],vmin=-1,vmax=1,cmap='magma');axs[0,c].axvline(.8,c='cyan',ls='--',lw=1);axs[0,c].set_title(str(g)+' / first window',fontsize=9);axs[0,c].set_xlabel('ACC context time (s); target ends at0.8s')
  for row,p in enumerate([va['y'][i]]+[preds[r]['audio'][i] for r in routes],1):axs[row,c].imshow(p,origin='lower',aspect='auto',extent=[0,.8,0,80],vmin=-1,vmax=1,cmap='magma');axs[row,c].set_xlabel('Target audio time (s)')
  tt=np.arange(81)*.01;axs[5,c].plot(tt,va['audio_activity'][i],c='black',label='Original audio activity');axs[5,c].plot(tt,va['activity'][i,20:101],c='grey',ls='--',label='TRAIN delay/release soft target')
  for r,color in zip(routes,['tab:blue','tab:orange','tab:green']):axs[5,c].plot(tt,preds[r]['activity'][i,20:101],c=color,label=r)
  axs[5,c].set_ylim(-.05,1.05);axs[5,c].set_xlabel('Audio time (s); ACC scores evaluated at t+200ms')
 for row,label in enumerate(['ACC magnitude (Hz)','Real microphone (Mel)','Temporal only (Mel)','Dual, no ACC loss (Mel)','Dual + ACC loss (Mel)','Soft activity score']):axs[row,0].set_ylabel(label)
 axs[5,0].legend(fontsize=6,loc='lower left');fig.suptitle('Fixed seed42, four unseen recordings; shared display scales\nACC-only inference; future context; identity preservation NOT established',fontsize=12);fig.savefig(ROOT/'validation_comparison.png',dpi=150);plt.close(fig)
 print(json.dumps(dict(controls=control_macro,summary=summary),indent=2))
if __name__=='__main__':main()
