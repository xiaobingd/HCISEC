"""Train-only global delay/release hypotheses; no validation-specific alignment.

This is reference-signal association analysis, not ACC-only authentication.
"""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy import signal,stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/delay_tail_20261018');OUT.mkdir(exist_ok=True)
cohort=json.loads(Path('outputs/raw_periodicity_20261017/source_and_direct_csv.json').read_text())
threshold=json.loads(Path('outputs/crossmodal_activity_20261017/results.json').read_text())['protocol']['audio_gmm']['threshold']
lags=np.arange(-.20,.201,.02);taus=np.array([0.,.03,.06,.10,.16,.25,.40])
def rms_trailing(x,w):
 p=np.r_[np.zeros((1,)+x.shape[1:]),np.cumsum(x*x,axis=0)];hi=np.arange(1,len(x)+1);lo=np.maximum(hi-w,0)
 return np.sqrt(np.maximum((p[hi]-p[lo])/(hi-lo).reshape((-1,)+(1,)*(x.ndim-1)),0))
def release(y,tau):
 if tau==0:return y.astype(float)
 v=np.zeros(len(y));decay=np.exp(-.01/tau)
 for i in range(len(y)):v[i]=max(float(y[i]),v[i-1]*decay if i else 0)
 return v
def response(r,d,tau,y=None):
 v=release(r['y'] if y is None else y,tau)
 return np.interp(r['t']-d,r['t'],v,left=v[0],right=v[-1])
records=[];names=['raw_'+k+'_20ms' for k in ['x','y','z','RSS']]
for split in ['train','val']:
 for g in cohort[split]:
  with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as z:x=z['acc_raw'].copy();a=z['audio'].copy();fs=int(z['audio_fs'])
  x-=x.mean(0);n=min(len(x)//10,int(len(a)/fs*100));t=np.arange(n)*.01
  ax=rms_trailing(x,20)[::10][:n];ar=rms_trailing(a,max(1,round(fs*.02)))[np.rint(t*fs).astype(int)]
  feat=np.column_stack([ax,np.sqrt(np.sum(ax*ax,axis=1))]);feat=np.log10(feat+1e-12)
  audio=np.log10(ar+1e-9);y=audio>threshold
  records.append(dict(id=g,split=split,t=t,x=feat,audio=audio,y=y.astype(float),keep=(t>=.6)&(t<t[-1]-.6)))
train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
def moments(r,v,j):
 k=r['keep'];x=r['x'][k,j];v=v[k];return np.array([np.mean(x),np.mean(v),np.mean(v*v),np.mean(v*x),np.mean(x*x)])
def fitmom(m):
 mx,mv,mvv,mvx,mxx=m;beta=max(0.,float((mvx-mv*mx)/max(mvv-mv*mv,1e-12)));return float(mx-beta*mv),beta
def loss(m,ab):
 mx,mv,mvv,mvx,mxx=m;a,b=ab;return float(mxx+a*a+b*b*mvv+2*a*b*mv-2*a*mx-2*b*mvx)
def fitgrid(j):
 candidates=[]
 for d in lags:
  for tau in taus:
   ms=np.array([moments(r,response(r,d,tau),j) for r in train]);total=ms.sum(0);scores=[]
   for i in range(len(train)):
    rest=(total-ms[i])/(len(train)-1);ab=fitmom(rest);constant=(rest[0],0.)
    baseline=loss(ms[i],constant);scores.append((baseline-loss(ms[i],ab))/max(baseline,1e-9))
   candidates.append(dict(delay_s=float(round(d,4)),release_tau_s=float(tau),train_loo_relative_mse_gain=float(np.mean(scores)),fit=fitmom(ms.mean(0))))
 return candidates
def pick(grid,which):
 subset=[c for c in grid if (which!='synchronous' or abs(c['delay_s'])<1e-6 and c['release_tau_s']==0) and (which!='delay_only' or c['release_tau_s']==0) and (which!='tail_only' or abs(c['delay_s'])<1e-6)]
 return max(subset,key=lambda c:c['train_loo_relative_mse_gain'])
def timeprofile(rr,t):
 return np.mean([np.interp(t,r['t'],r['y'],left=r['y'][0],right=r['y'][-1]) for r in rr],axis=0)
results={};grids={}
for j,name in enumerate(names):
 grid=fitgrid(j);grids[name]=grid;result={}
 for mode in ['synchronous','delay_only','tail_only','delay_and_tail']:
  c=pick(grid,mode);d,tau=c['delay_s'],c['release_tau_s'];ab=c['fit']
  # No-ACC/no-validation-audio elapsed-time control. TRAIN fitting uses LOO profiles.
  tm=np.array([moments(r,response(r,d,tau,timeprofile([z for z in train if z is not r],r['t'])),j) for r in train]);tab=fitmom(tm.mean(0))
  mean=np.mean([r['x'][r['keep'],j].mean() for r in train]);per=[]
  for r in val:
   k=r['keep'];x=r['x'][k,j];v=response(r,d,tau);pred=ab[0]+ab[1]*v[k];base=np.mean((x-mean)**2)
   tv=response(r,d,tau,timeprofile(train,r['t']));tp=tab[0]+tab[1]*tv[k]
   wrong=[]
   for other in val:
    if other is r:continue
    vv=response(r,d,tau,np.interp(r['t'],other['t'],other['y']));wp=ab[0]+ab[1]*vv[k];wrong.append(np.mean((wp-x)**2))
   corr=float(stats.spearmanr(x,v[k]).statistic) if np.std(v[k])>1e-8 else None
   per.append(dict(recording=r['id'],normalized_mse=float(np.mean((pred-x)**2)/base),time_only_normalized_mse=float(np.mean((tp-x)**2)/base),wrong_record_normalized_mse=float(np.mean(wrong)/base),matched_minus_wrong_mse_gain=float((np.mean(wrong)-np.mean((pred-x)**2))/base),spearman=corr))
  result[mode]=dict(**c,per_record=per,macro={k:float(np.mean([r[k] for r in per if r[k] is not None])) for k in per[0] if k!='recording'})
 results[name]=result;print(name,{k:(v['delay_s'],v['release_tau_s'],v['train_loo_relative_mse_gain'],v['macro']) for k,v in result.items()},flush=True)
# Global feature choice TRAIN only. Primary RSS remains reported independently.
selected=max(names,key=lambda name:results[name]['delay_and_tail']['train_loo_relative_mse_gain'])
def events(r,kind):
 y=r['y'].astype(bool);indices=np.flatnonzero(np.diff(y.astype(int))==(1 if kind=='onset' else -1))+1;out=[]
 for i in indices:
  if i<60 or i+60>=len(y):continue
  if kind=='offset' and (not np.all(y[i-12:i]) or np.any(y[i:i+60])):continue
  if kind=='onset' and (np.any(y[i-20:i]) or not np.all(y[i:i+12])):continue
  out.append(int(i))
 return out
event_results={}
for split,rr in [('train',train),('val',val)]:
 event_results[split]={}
 for kind in ['onset','offset']:
  per=[]
  for r in rr:
   ee=events(r,kind);per.append(dict(recording=r['id'],events=len(ee),indices=ee))
  event_results[split][kind]=per
protocol=dict(source='same20train4reusedval whole recordings, one person; test and28reserve unread',primary='raw DCremoved XYZ andRSS, common20ms TRAILING RMS; no frequency filter or250ms smoothing to manufacture tail',frames_ms=10,edge_exclusion_s=.6,delay_search_s=[-.2,.2,.02],positive_delay='ACC envelope response after microphone activity',release_taus_s=taus.tolist(),response='instant attack, exponential release of binary audio activity; nonnegative slope + intercept fit to ACC logRMS; phenomenological, NOT tissue mechanical timeconstant',selection='20fold leave-one-record-out TRAIN normalizedMSE gain, record-balanced, global settings, no per-validation-record shift or tail fit',audio_threshold=threshold,models='synchronous; trainselecteddelayonly; trainselectedtailonly; trainselecteddelay+tail',controls='same elapsed time TRAIN activity profile, other VALIDATION recording activity with fixed fitted response; no circular wrap',scope='audio-reference association diagnostic, not ACC-only inference/authentication; lowfreq headmotion and hardware filtering remain confounds',event_rule='offset >=120ms active before >=600ms quiet after; onset >=200ms quiet before >=120ms active after; full +/-600ms support; allfixed across records. Quiet criterion tightened for event plot only to exclude renewed speech, not used in lag fitting.',statistics='recordings are units; event counts reported; no frame-independence significance; descriptive4val, no identity metric')
summary=dict(protocol=protocol,train_selected_feature=selected,primary_feature='raw_RSS_20ms',features=results,event_counts=event_results)
(OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'train_search.json').write_text(json.dumps(grids,indent=2))
fig,axs=plt.subplots(4,1,figsize=(13,11),layout='constrained');j=names.index(selected);c=results[selected]['delay_and_tail'];mu=np.mean([r['x'][r['keep'],j].mean() for r in train]);sd=np.sqrt(np.mean([np.mean((r['x'][r['keep'],j]-mu)**2) for r in train]))
for ax,r in zip(axs,val):
 ax.plot(r['t'],(r['x'][:,j]-mu)/sd,color='tab:blue',label='ACC log RMS, shared TRAIN scale (left)');ax.set_ylabel('ACC z score');ax.set_ylim(-4,4);ax.set_title(r['id']+' / '+selected,fontsize=10)
 right=ax.twinx();right.plot(r['t'],r['y'],color='black',lw=1,label='Audio activity reference');right.plot(r['t'],response(r,c['delay_s'],c['release_tau_s']),color='tab:orange',label='Fixed TRAIN delay + release');right.set_ylim(-.05,1.1);right.set_ylabel('Activity / response')
 if r is val[0]:ax.legend(fontsize=8,loc='lower left');right.legend(fontsize=8,loc='upper right')
axs[-1].set_xlabel('Elapsed recording time (s)');fig.suptitle(f"Train-only selected delay {c['delay_s']*1000:.0f}ms, release tau {c['release_tau_s']*1000:.0f}ms\nNo per-record alignment; same20ms trailing RMS; reference association, not identity verification",fontsize=12);fig.savefig(OUT/'delay_overlay.png',dpi=130);plt.close(fig)
dt=np.arange(-60,61)*.01
for feature in [selected,'raw_RSS_20ms']:
 j=names.index(feature);mu=np.mean([r['x'][r['keep'],j].mean() for r in train]);sd=np.sqrt(np.mean([np.mean((r['x'][r['keep'],j]-mu)**2) for r in train]));fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for row,(split,rr) in enumerate([('train',train),('val',val)]):
  for col,kind in enumerate(['onset','offset']):
   curves=[];audios=[];ne=0
   for r in rr:
    ee=events(r,kind)
    if not ee:continue
    curves.append(np.mean([((r['x'][i-60:i+61,j]-mu)/sd) for i in ee],axis=0));audios.append(np.mean([r['y'][i-60:i+61] for i in ee],axis=0));ne+=len(ee)
   ax=axs[row,col];ax.axvline(0,color='gray',ls='--');ax.set_title(f'{split} {kind}: {ne} events / {len(curves)} recordings')
   if curves:
    for curve in curves:ax.plot(dt,curve,color='tab:blue',alpha=.13,lw=.7)
    ax.plot(dt,np.mean(curves,axis=0),color='tab:blue',lw=2,label='ACC mean (left)');right=ax.twinx();right.plot(dt,np.mean(audios,axis=0),color='black',label='Audio activity (right)');right.set_ylim(-.05,1.05)
    if row==0 and col==0:ax.legend(fontsize=8,loc='upper left');right.legend(fontsize=8,loc='lower right')
   ax.set_ylim(-2,2);ax.set_xlabel('Time relative to audio event (s)');ax.set_ylabel('ACC shared TRAIN z score')
 fig.suptitle(feature+': event-locked mean and faint individual recordings\nOffset requires600ms subsequent quiet; no per-event normalization or mechanical-decay claim');fig.savefig(OUT/('event_locked.png' if feature==selected else 'event_locked_rss.png'),dpi=140);plt.close(fig)
print('DONE',OUT,'selected',selected,flush=True)
