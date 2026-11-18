"""Compare RAW generated Mel and retained ACC cycle, with fixed shared scales."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/cycle_audio_domain_20261018')
def main():
 with np.load('outputs/activity_alignment_20261018/data_val.npz') as d:va={k:d[k] for k in d.files}
 with np.load('outputs/activity_alignment_20261018/data_train.npz') as d:audio_mean=d['y'].mean(0);acc_mean=d['x'].mean(0)
 control_audio=float(np.mean([np.mean(abs(va['y'][va['ids']==g]-audio_mean)) for g in np.unique(va['ids'])]));control_acc=float(np.mean([np.mean(abs(va['x'][va['ids']==g]-acc_mean)) for g in np.unique(va['ids'])]))
 runs=json.loads((ROOT/'results.json').read_text());routes=['continuation','audio_anchor','audio_anchor_smooth'];summary={}
 for route in routes:
  rr=[r for r in runs if r['route']==route];summary[route]={k:dict(mean=float(np.mean([r['macro'][k] for r in rr])),std=float(np.std([r['macro'][k] for r in rr]))) for k in rr[0]['macro']}
 public=dict(scope='same personal20train4reusedval,58/16windows;6fixedendpointfine-tunes2seeds; noidentitymetrics',controls=dict(mean_audio_mae=control_audio,mean_ACC_mae=control_acc),summary=summary,runs=runs)
 (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
 pred={r:dict(np.load(ROOT/(r+'_42_predictions.npz'))) for r in routes};initial=dict(np.load(ROOT/'initial_42_predictions.npz'));ug=np.unique(va['ids']);fig,axs=plt.subplots(7,4,figsize=(17,17),layout='constrained')
 for c,g in enumerate(ug):
  i=np.flatnonzero(va['ids']==g)[0];arrays=[va['y'][i],initial['generated_audio'][i]]+[pred[r]['generated_audio'][i] for r in routes]+[va['x'][i],pred['audio_anchor_smooth']['cycle_ACC'][i]]
  for row,a in enumerate(arrays):
   acc=row>=5;image=a[:195].reshape(3,65,81).mean(0) if acc else a;axs[row,c].imshow(image,origin='lower',aspect='auto',extent=[0,.8,0,500 if acc else 80],vmin=-1,vmax=1,cmap='magma');axs[row,c].set_xlabel('Time (s)')
  axs[0,c].set_title(str(g)+' / first frozen window',fontsize=9)
 for row,label in enumerate(['Real microphone (Mel)','Initial cycle-priority G (Mel)','Matched continuation G (Mel)','Audio-anchored G (Mel)','Audio-anchored + S: RAW G (Mel)','Real ACC XYZ mean (Hz)','Cycle ACC with S (Hz)']):axs[row,0].set_ylabel(label)
 fig.suptitle('Full cycle audio-domain repair: fixed seed42, four unseen recordings\nRaw G outputs, same scales, no cosmetic smoothing of generated audio; no identity claim',fontsize=12);fig.savefig(ROOT/'audio_comparison.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained');xx=np.arange(3);labels=['Continuation','Audio anchor','Audio anchor + S']
 axs[0].bar(xx,[summary[r]['audio_mae']['mean'] for r in routes],yerr=[summary[r]['audio_mae']['std'] for r in routes],capsize=4);axs[0].axhline(control_audio,c='black',ls='--',label='Train mean audio');axs[0].set_title('RAW generated audio MAE');axs[0].legend(fontsize=8)
 axs[1].bar(xx,[summary[r]['ACC_cycle_mae']['mean'] for r in routes],yerr=[summary[r]['ACC_cycle_mae']['std'] for r in routes],capsize=4);axs[1].axhline(control_acc,c='black',ls='--',label='Train mean ACC');axs[1].set_title('ACC cycle MAE');axs[1].legend(fontsize=8)
 axs[2].bar(xx,[summary[r]['audio_spectral_second_difference_ratio']['mean'] for r in routes]);axs[2].axhline(1,c='black',ls='--',label='Real audio');axs[2].set_title('Frequency curvature / real audio');axs[2].legend(fontsize=8)
 for ax in axs:ax.set_xticks(xx,labels,rotation=12)
 fig.suptitle('Two-seed means +/- SD; repeated exploratory validation, no identity verification',fontsize=11);fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
 # Tiny readable selection is fixed first recording, not a best-looking example.
 i=0;fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
 for ax,(label,a) in zip(axs.flat,[('Real microphone',va['y'][i]),('Initial cycle model',initial['generated_audio'][i]),('Continuation control',pred['continuation']['generated_audio'][i]),('Audio anchor',pred['audio_anchor']['generated_audio'][i]),('Audio anchor + S (RAW G)',pred['audio_anchor_smooth']['generated_audio'][i]),('Cycle ACC, audio anchor + S',pred['audio_anchor_smooth']['cycle_ACC'][i,:195].reshape(3,65,81).mean(0))]):
  ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(label,fontsize=10);ax.set_xlabel('Time frame');ax.set_ylabel('Mel bin' if 'Cycle ACC' not in label else 'ACC frequency bin')
 fig.suptitle(str(va['ids'][i])+' / first frozen window / seed42 / shared scales',fontsize=11);fig.savefig(ROOT/'compact_comparison.png',dpi=150);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
