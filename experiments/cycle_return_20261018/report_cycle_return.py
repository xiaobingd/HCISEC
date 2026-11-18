"""Prioritize full-cycle preservation, compare template and bottleneck controls."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/cycle_return_20261018')
def main():
 with np.load('outputs/activity_alignment_20261018/data_val.npz') as d:va={k:d[k] for k in d.files}
 runs=json.loads((ROOT/'results.json').read_text());routes=['cycle_balanced','acc_priority','robust_priority'];summary={}
 for route in routes:
  rr=[r for r in runs if r['route']==route];summary[route]={k:dict(mean=float(np.mean([r['macro'][k] for r in rr])),std=float(np.std([r['macro'][k] for r in rr]))) for k in rr[0]['macro']}
 public=dict(scope='oneperson;20train4REUSEDval,58/16 windows;4test+28reserveUNREAD; newfullbidirectionalcycle, noidentityverification',summary=summary,runs=runs)
 (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
 preds={r:dict(np.load(ROOT/(r+'_42_predictions.npz'))) for r in routes};ug=np.unique(va['ids']);fig,axs=plt.subplots(7,4,figsize=(17,17),layout='constrained')
 for c,g in enumerate(ug):
  i=np.flatnonzero(va['ids']==g)[0]
  arrays=[va['x'][i]]+[preds[r]['cycle_ACC'][i] for r in routes]+[va['y'][i],preds['robust_priority']['generated_audio'][i],preds['robust_priority']['cycle_quant8'][i]]
  for row,a in enumerate(arrays):
   acc=row not in [4,5];image=a[:195].reshape(3,65,81).mean(0) if acc else a;axs[row,c].imshow(image,origin='lower',aspect='auto',extent=[0,.8,0,500 if acc else 80],vmin=-1,vmax=1,cmap='magma');axs[row,c].set_xlabel('Time (s)')
  axs[0,c].set_title(str(g)+' / first frozen window',fontsize=9)
 for row,label in enumerate(['Real ACC XYZ mean (Hz)','Balanced ACC cycle (Hz)','Priority ACC cycle (Hz)','Robust ACC cycle (Hz)','Real microphone (Mel)','Robust G(ACC) audio (Mel)','Robust F(quant8(G(ACC))) (Hz)']):axs[row,0].set_ylabel(label)
 fig.suptitle('ACC -> generated audio -> ACC: fixed seed42, four unseen recordings\nNo latent bypass between generators; shared scales; no identity claim',fontsize=12);fig.savefig(ROOT/'cycle_comparison.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained');xx=np.arange(3);labels=['Balanced','ACC priority','Robust priority']
 axs[0].bar(xx,[summary[r]['ACC_cycle_mae']['mean'] for r in routes],yerr=[summary[r]['ACC_cycle_mae']['std'] for r in routes],capsize=4);axs[0].axhline(summary[routes[0]]['ACC_template_mae']['mean'],ls='--',c='black',label='Train mean ACC');axs[0].set_title('ACC cycle MAE (lower better)');axs[0].legend(fontsize=8)
 axs[1].bar(xx,[summary[r]['ACC_magnitude_detail_corr']['mean'] for r in routes]);axs[1].axhline(summary[routes[0]]['ACC_template_detail_corr']['mean'],ls='--',c='black',label='Train mean ACC');axs[1].set_title('ACC double-centered detail correlation');axs[1].legend(fontsize=8)
 for j,key in enumerate(['quant8','noise005','blur3']):axs[2].bar(xx+(j-1)*.23,[summary[r][key+'_cycle_mae_increase']['mean'] for r in routes],width=.23,label=key)
 axs[2].set_title('Cycle MAE increase after Mel perturbation');axs[2].legend(fontsize=8)
 for ax in axs:ax.set_xticks(xx,labels,rotation=12)
 fig.suptitle('Full cycle preservation: mean +/- SD of two seeds; four reused validation recordings, not authentication',fontsize=11);fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(4,1,figsize=(12,9),layout='constrained');t=np.arange(81)*.01
 for ax,g in zip(axs,ug):
  i=np.flatnonzero(va['ids']==g)[0];ax.plot(t,va['x'][i,198],c='black',label='Real ACC raw-RSS50ms, shared TRAIN scaling')
  for r,color in zip(routes,['tab:blue','tab:orange','tab:green']):ax.plot(t,preds[r]['cycle_ACC'][i,198],c=color,label=r)
  ax.set_title(str(g)+' / first window',fontsize=9);ax.set_ylabel('Log RMS z score')
 axs[0].legend(fontsize=7);axs[-1].set_xlabel('Time (s)');fig.suptitle('An ACC energy feature through the complete cycle, same scale for every recording');fig.savefig(ROOT/'cycle_energy.png',dpi=140);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
