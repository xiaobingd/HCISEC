"""Keep paired prewarm separate from FINAL complete-cycle results."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/cycle_audio_domain_20261018');RES=ROOT/'resnet_results'
def summarize(rr):return {k:dict(mean=float(np.mean([r['macro'][k] for r in rr])),std=float(np.std([r['macro'][k] for r in rr]))) for k in rr[0]['macro']}
def main():
 runs=json.loads((RES/'results.json').read_text());prewarm=json.loads((RES/'prewarm_results.json').read_text());summary=dict(final_full_cycle=summarize(runs),paired_prewarm_only=summarize(prewarm))
 with np.load('outputs/activity_alignment_20261018/data_val.npz') as d:va={k:d[k] for k in d.files}
 final=dict(np.load(RES/'resnet_curriculum_42_predictions.npz'));pre=dict(np.load(RES/'resnet_curriculum_42_prewarm_predictions.npz'));initial=dict(np.load(ROOT/'initial_42_predictions.npz'))
 public=dict(scope='same oneperson20train4REUSEDval58/16;singlearchitecture+curriculum exploration2seeds; noidentityverification; finalstagecompletecycle, prewarmaloneNOTcycle; longerbudget thanwarmstartedtemporal',summary=summary,runs=runs,prewarm_runs=prewarm)
 (RES/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
 ug=np.unique(va['ids']);fig,axs=plt.subplots(6,4,figsize=(17,15),layout='constrained')
 for c,g in enumerate(ug):
  i=np.flatnonzero(va['ids']==g)[0];arrays=[va['y'][i],initial['generated_audio'][i],pre['generated_audio'][i],final['generated_audio'][i],va['x'][i],final['cycle_ACC'][i]]
  for row,a in enumerate(arrays):
   acc=row>=4;image=a[:195].reshape(3,65,81).mean(0) if acc else a;axs[row,c].imshow(image,origin='lower',aspect='auto',extent=[0,.8,0,500 if acc else 80],vmin=-1,vmax=1,cmap='magma');axs[row,c].set_xlabel('Time (s)')
  axs[0,c].set_title(str(g)+' / first window',fontsize=9)
 for row,label in enumerate(['Real microphone (Mel)','Previous cycle model G (Mel)','ResNet PAIRED PREWARM only','ResNet FINAL FULL CYCLE G','Real ACC XYZ mean (Hz)','FINAL full-cycle ACC (Hz)']):axs[row,0].set_ylabel(label)
 fig.suptitle('Frequency-aware ResNet + real-pair curriculum: seed42, fixed samples and shared scales\nPrewarm is not a cycle result; FINAL is a complete bidirectional cycle; no identity claim',fontsize=12);fig.savefig(RES/'full_comparison.png',dpi=140);plt.close(fig)
 i=0;fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained');arrays=[('Real microphone',va['y'][i]),('Previous cycle G',initial['generated_audio'][i]),('Paired PREWARM only',pre['generated_audio'][i]),('FINAL full-cycle G',final['generated_audio'][i]),('Real ACC',va['x'][i,:195].reshape(3,65,81).mean(0)),('FINAL cycle ACC',final['cycle_ACC'][i,:195].reshape(3,65,81).mean(0))]
 for ax,(name,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(name);ax.set_xlabel('Time frame');ax.set_ylabel('Frequency bin')
 fig.suptitle(str(va['ids'][i])+' / first window / fixed seed42 / shared scales');fig.savefig(RES/'compact_comparison.png',dpi=150);plt.close(fig)
 if (ROOT/'training_examples.npz').exists():
  train=dict(np.load(ROOT/'training_examples.npz'));fig,axs=plt.subplots(4,4,figsize=(14,10),layout='constrained')
  for c,name in enumerate(ug):
   i=np.flatnonzero(va['ids']==name)[0]
   for row,a in enumerate([train['real_audio'][c],train['generated_audio'][c],va['y'][i],final['generated_audio'][i]]):axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');axs[row,c].set_xlabel('Time frame')
   axs[0,c].set_title('TRAIN '+str(train['ids'][c]),fontsize=8);axs[2,c].set_title('VAL '+str(name),fontsize=8)
  for row,label in enumerate(['TRAIN real','TRAIN generated','VAL real','VAL generated']):axs[row,0].set_ylabel(label+' / Mel bin')
  fig.suptitle('Same FINAL full-cycle ResNet, seed42; first TRAIN and VAL recordings, different clips\nShared scales; examples fixed by recording order, not selected by reconstruction quality',fontsize=11);fig.savefig(RES/'train_val_comparison.png',dpi=150);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
