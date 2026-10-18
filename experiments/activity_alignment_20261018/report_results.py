"""Frozen-window controls and honest visual report, no held-out test access."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/activity_alignment_20261018')
def auc(y,p):
 y=np.asarray(y).ravel().astype(bool);p=np.asarray(p).ravel();n=y.sum();m=len(y)-n
 return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n*m else None
def corr(a,b):return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
def detail(y):return y-y.mean(-2,keepdims=True)-y.mean(-1,keepdims=True)+y.mean((-2,-1),keepdims=True)
def main():
 tr=dict(np.load(ROOT/'data_train.npz'));va=dict(np.load(ROOT/'data_val.npz'));results=json.loads((ROOT/'results.json').read_text());pred={r['route']:dict(np.load(ROOT/(r['route']+'_predictions.npz'))) for r in results}
 ids=va['ids'];groups=np.unique(ids);rows=json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text())['samples']
 # Time-only controls fitted solely on TRAIN, record-balanced for elapsed-time profile.
 grid=np.arange(int(np.ceil(max(r['start_s'] for r in rows if r['split']=='train')*100))+83)*.01;profiles=[]
 for g in np.unique(tr['ids']):
  accum=np.zeros(len(grid));count=np.zeros(len(grid))
  trainrows=[r for r in rows if r['split']=='train'];ii=np.flatnonzero(tr['ids']==g)
  for i in ii:
   bins=np.rint((trainrows[i]['start_s']+np.arange(81)*.01)*100).astype(int);np.add.at(accum,bins,tr['activity'][i]);np.add.at(count,bins,1)
  profiles.append(np.where(count>0,accum/np.maximum(count,1),np.nan))
 profiles=np.array(profiles);counts=np.isfinite(profiles).sum(0);profile=np.nansum(profiles,0)/np.maximum(counts,1);known=counts>0
 validrows=[r for r in rows if r['split']=='val'];ptime=np.array([np.interp(r['start_s']+np.arange(81)*.01,grid[known],profile[known]) for r in validrows]);prel=np.broadcast_to(tr['activity'].mean(0),(len(ids),81))
 # Simple raw-input logistic control: fixed L2 coefficient .01, equal recording contribution.
 x=tr['x'].transpose(0,2,1).reshape(-1,251).astype('float64');y=tr['activity'].ravel();mu=x.mean(0);sd=np.maximum(x.std(0),.05);x=(x-mu)/sd
 weights=np.concatenate([np.full(81,1/np.sum(tr['ids']==g)) for g in tr['ids']]);weights/=weights.sum()
 def objective(w):
  logits=x@w[:-1]+w[-1];prob=1/(1+np.exp(-np.clip(logits,-40,40)));v=np.sum(weights*(np.logaddexp(0,logits)-y*logits))+.01*np.sum(w[:-1]**2)
  grad=weights*(prob-y);return v,np.r_[x.T@grad+.02*w[:-1],grad.sum()]
 fit=minimize(objective,np.zeros(252),jac=True,method='L-BFGS-B',options={'maxiter':250});vx=va['x'].transpose(0,2,1).reshape(-1,251);logits=((vx-mu)/sd)@fit.x[:-1]+fit.x[-1];plog=(1/(1+np.exp(-np.clip(logits,-40,40)))).reshape(len(ids),81)
 controls=[]
 template=tr['y'].mean(0)
 for g in groups:
  ix=ids==g;yy=va['y'][ix];pp=np.broadcast_to(template,yy.shape)
  controls.append(dict(recording=str(g),time_elapsed_activity_auc=auc(va['activity'][ix],ptime[ix]),time_relative_activity_auc=auc(va['activity'][ix],prel[ix]),raw_logistic_activity_auc=auc(va['activity'][ix],plog[ix]),template_audio_mae=float(abs(yy-pp).mean()),template_detail_corr=corr(detail(yy),detail(pp))))
 cm={k:float(np.mean([r[k] for r in controls])) for k in controls[0] if k!='recording'}
 summary=dict(controls_macro=cm,controls_per_record=controls,raw_logistic_converged=bool(fit.success),routes=results,scope='single-person exploratory validation; activity AUC is not identity AUC; neural activity head in reconstruction route is untrained and must not be interpreted')
 (ROOT/'summary_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 first=[np.flatnonzero(ids==g)[0] for g in groups];nrow=3+len(results);fig,axs=plt.subplots(nrow,4,figsize=(17,2.5*nrow),layout='constrained');t=np.arange(81)*.01
 for c,(g,i) in enumerate(zip(groups,first)):
  acc=va['x'][i,:195].reshape(3,65,81).mean(0)
  axs[0,c].imshow(acc,origin='lower',aspect='auto',extent=[0,.8,0,500],vmin=-1,vmax=1,cmap='magma');axs[0,c].set_title(str(g)+' / first frozen window',fontsize=10)
  arrays=[va['y'][i]]+[pred[r['route']]['audio'][i] for r in results]
  for row,a in enumerate(arrays,1):axs[row,c].imshow(a,origin='lower',aspect='auto',extent=[0,.8,0,80],vmin=-1,vmax=1,cmap='magma')
  axs[-1,c].plot(t,va['activity'][i],color='black',lw=1,label='Audio energy label (reference)')
  for r,color in zip(results[1:],['tab:orange','tab:green','tab:blue']):axs[-1,c].plot(t,pred[r['route']]['activity'][i],label=r['route'],color=color)
  axs[-1,c].plot(t,ptime[i],color='grey',ls='--',label='Elapsed-time-only control');axs[-1,c].set_ylim(-.05,1.05)
  for row in range(nrow):axs[row,c].set_xlabel('Time within window (s)')
 labels=['Real ACC XYZ mean spectrum (Hz)','Real microphone (Mel bin)','Reconstruction baseline (Mel bin)','Activity + alignment (Mel bin)','+ ACC preservation (Mel bin)','Frozen ACC encoder (Mel bin)','Activity score'] if len(results)==4 else ['Real ACC XYZ mean spectrum (Hz)','Real microphone (Mel bin)','Reconstruction baseline (Mel bin)','Activity + alignment (Mel bin)','+ ACC preservation (Mel bin)','Activity score']
 for row,label in enumerate(labels):axs[row,0].set_ylabel(label)
 axs[-1,0].legend(fontsize=6,loc='lower left');fig.suptitle('Audio-supervised ACC learning: fixed unseen recordings, shared display scales\nOne participant; no inference microphone; repeated exploratory validation, not identity verification',fontsize=13)
 fig.savefig(ROOT/'validation_comparison.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained');names=['Reconstruction','Activity + align','+ ACC preserve','Frozen encoder'][:len(results)];xx=np.arange(len(results))
 axs[0].bar(xx,[r['macro']['audio_mae'] for r in results]);axs[0].axhline(cm['template_audio_mae'],ls='--',color='black',label='Train mean template');axs[0].set_title('Audio MAE (lower better)');axs[0].legend(fontsize=8)
 for j,key in enumerate(['spectrum','energy']):axs[1].bar(xx+(j-.5)*.3,[r['acc_retention_probe'][key+'_normalized_mse'] for r in results],width=.3,label=key)
 axs[1].axhline(1,ls='--',color='black');axs[1].set_title('ACC ridge recovery error / mean baseline');axs[1].legend(fontsize=8)
 axs[2].bar(xx[1:],[r['macro']['activity_auc'] for r in results[1:]],color=['tab:orange','tab:green','tab:blue'][:len(results)-1]);axs[2].axhline(cm['time_elapsed_activity_auc'],ls='--',color='grey',label='Elapsed time only');axs[2].axhline(cm['raw_logistic_activity_auc'],ls=':',color='black',label='Raw ACC logistic');axs[2].set_ylim(0,1);axs[2].set_title('Audio activity AUC, not identity AUC');axs[2].legend(fontsize=8)
 for ax in axs:ax.set_xticks(xx,names,rotation=15,fontsize=8)
 fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
