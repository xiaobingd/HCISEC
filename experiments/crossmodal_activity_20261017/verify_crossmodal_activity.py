"""Test cross-modal ACTIVITY correspondence; no same-frequency assumption or generator."""
import json
from pathlib import Path
import numpy as np
from scipy import signal,ndimage,optimize,special,stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('outputs/crossmodal_activity_20261017');OUT.mkdir(exist_ok=True)
cohort=json.loads(Path('outputs/raw_periodicity_20261017/source_and_direct_csv.json').read_text())
bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)]
names=[];records=[]
for split in ('train','val'):
 for g in cohort[split]:
  with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as d:x=d['acc_raw'].copy();a=d['audio'].copy();fs=int(d['audio_fs'])
  x=x-x.mean(0);n=min(len(x)//10,len(a)//(fs//100));t=np.arange(n)*.01
  audio_rms=np.sqrt(ndimage.uniform_filter1d(a*a,size=int(fs*.02),mode='reflect'))[::fs//100][:n]
  channels=[];channelnames=[]
  for label,xx in [('raw_dc',x)]+[(f'{lo}-{hi}Hz',signal.sosfiltfilt(signal.butter(4,[lo,hi],btype='bandpass',fs=1000,output='sos'),x,axis=0)) for lo,hi in bands]:
   for ms in (50,250):
    power=ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect')[::10][:n]
    for k,key in enumerate('xyz'):
     channels.append(np.log10(np.sqrt(np.maximum(power[:,k],0))+1e-12));channelnames.append(f'{label}_{key}_{ms}ms')
    channels.append(np.log10(np.sqrt(np.maximum(power.sum(1),0))+1e-12));channelnames.append(f'{label}_RSS_{ms}ms')
  if not names:names=channelnames
  assert names==channelnames
  keep=(t>=.25)&(t<min(len(x)/1000,len(a)/fs)-.25)
  records.append(dict(id=g,split=split,t=t[keep],x=np.array(channels).T[keep],audio=np.log10(audio_rms[keep]+1e-9)))

# One shared monotonic audio activity threshold, learned using TRAIN audio only.
train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
audio_all=np.concatenate([r['audio'] for r in train])
# PCM16 quantization: a nonzero20ms frame has RMS >=~1.7e-6.
# Fit quiet vs active on genuinely nonzero frames, rather than digital-zero vs everything.
audio=audio_all[audio_all>-6.];means=np.quantile(audio,[.2,.8]);var=np.full(2,np.var(audio));mix=np.array([.5,.5])
for _ in range(150):
 logp=-.5*((audio[:,None]-means)**2/var+np.log(2*np.pi*var))+np.log(mix)
 resp=np.exp(logp-special.logsumexp(logp,axis=1,keepdims=True));total=resp.sum(0)
 means=(resp*audio[:,None]).sum(0)/total;var=np.maximum((resp*(audio[:,None]-means)**2).sum(0)/total,.0025);mix=total/len(audio)
order=np.argsort(means);means,var,mix=means[order],var[order],mix[order]
g=np.linspace(*means,10001);logp=-.5*((g[:,None]-means)**2/var+np.log(2*np.pi*var))+np.log(mix);threshold=float(g[np.argmin(abs(logp[:,0]-logp[:,1]))])
for r in records:r['y']=(r['audio']>threshold).astype(int)
active_times=np.concatenate([r['t'][r['y']==1] for r in train]);active_span=np.quantile(active_times,[.01,.99])
print('AUDIO GMM',means,var,mix,'threshold',threshold,flush=True)

def auc(y,p):
 y=np.asarray(y);n1=y.sum();n0=len(y)-n1
 if not n1 or not n0:return None
 ranks=stats.rankdata(p);return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def metrics(y,p,cut):
 v=p>=cut;tp=np.sum(v&(y==1));tn=np.sum(~v&(y==0));fp=np.sum(v&(y==0));fn=np.sum(~v&(y==1))
 return dict(activity_auc=auc(y,p),activity_f1=float(2*tp/max(2*tp+fp+fn,1)),balanced_accuracy=float(.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))),positive_fraction=float(y.mean()),frames=len(y))
def record_macro(seq):
 keys=['activity_auc','activity_f1','balanced_accuracy']
 return {k:float(np.mean([s[k] for s in seq if s[k] is not None])) for k in keys}

def fit(rr):
 X=np.concatenate([r['x'] for r in rr]);y=np.concatenate([r['y'] for r in rr]);mu=X.mean(0);sd=np.maximum(X.std(0),.05);z=(X-mu)/sd
 weights=[]
 for r in rr:
  yy=r['y'];n1=max(yy.sum(),1);n0=max((1-yy).sum(),1);weights.extend(np.where(yy==1,.5/n1,.5/n0)/len(rr))
 w=np.array(weights);w/=w.sum()
 def objective(b):
  h=z@b[:-1]+b[-1];loss=np.sum(w*(np.logaddexp(0,h)-y*h))+.005*np.sum(b[:-1]**2)
  e=w*(special.expit(h)-y);grad=np.r_[z.T@e+.01*b[:-1],e.sum()]
  return loss,grad
 res=optimize.minimize(objective,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
 assert np.isfinite(res.fun)
 return mu,sd,res.x
def predict(model,r):
 mu,sd,b=model;return special.expit(((r['x']-mu)/sd)@b[:-1]+b[-1])
def time_profile(rr,r):
 # No ACC: controls for repeated utterances at similar elapsed recording times.
 idx=np.rint(np.concatenate([z['t'] for z in rr])*100).astype(int);y=np.concatenate([z['y'] for z in rr]);num=np.bincount(idx,weights=y);den=np.bincount(idx)
 num=ndimage.gaussian_filter1d(num.astype(float),5);den=ndimage.gaussian_filter1d(den.astype(float),5)
 prior=np.divide(num,np.maximum(den,1e-9));globalprior=float(y.mean());where=np.rint(r['t']*100).astype(int)
 return np.array([prior[i] if i<len(prior) and den[i]>.01 else globalprior for i in where])

oof=[];time_oof=[]
for i,r in enumerate(train):
 rest=[s for s in train if s['id']!=r['id']];model=fit(rest);oof.append(predict(model,r));time_oof.append(time_profile(rest,r))
def choose_cut(predictions):
 cuts=np.linspace(.05,.95,91);scores=[np.mean([metrics(r['y'],p,c)['balanced_accuracy'] for r,p in zip(train,predictions)]) for c in cuts]
 return float(cuts[np.argmax(scores)])
cut=choose_cut(oof);tcut=choose_cut(time_oof);model=fit(train)
np.savez_compressed(OUT/'activity_probe.npz',mean=model[0],std=model[1],coefficients=model[2],threshold=cut,audio_threshold=threshold,feature_names=np.array(names))

# Describe ALL fixed features; a single feature/direction may be selected on TRAIN only.
features={}
for j,name in enumerate(names):
 tr=[auc(r['y'],r['x'][:,j]) for r in train];tm=float(np.mean([x for x in tr if x is not None]));direction=1 if tm>=.5 else -1
 per={}
 for r in val:
  per[r['id']]=dict(activity_auc=auc(r['y'],direction*r['x'][:,j]),spearman_audio_energy=float(stats.spearmanr(r['audio'],r['x'][:,j]).statistic),speech_minus_quiet_log10_rms=float(r['x'][r['y']==1,j].mean()-r['x'][r['y']==0,j].mean()))
 features[name]=dict(train_raw_auc=tm,train_selected_direction=direction,validation_macro_auc=float(np.mean([x['activity_auc'] for x in per.values() if x['activity_auc'] is not None])),validation_per_record=per)
selected=max(features,key=lambda n:abs(features[n]['train_raw_auc']-.5))
result=dict(protocol=dict(source='same personal20train4val recordings, whole common-duration portions',goal='cross-modal vocal/audio activity correspondence, not identical F0 or spectra',audio_labels='one monotonic2Gaussian threshold fitted on pooledTRAIN NONZERO20ms audio logRMS>-6; no per-record thresholds; automatic energetic activity surrogate, not manually verified speech',audio_gmm=dict(means=means.tolist(),variances=var.tolist(),mixtures=mix.tolist(),threshold=threshold),train_active_time_span=active_span.tolist(),bands=bands,acc_features='signed XYZ band energy plus RSS after power,50/250ms RMS, global TRAIN-only calibration',edge_exclusion_ms=250,frames='10ms, eachrecordonce; temporal frames are correlated,4validationrecordings independent units',probe='L2 logistic, balancedclasses AND recordings, fixed regularization. Leave-one-record-out TRAIN-only scaling/fits and threshold selection',time_baseline='smoothedTRAINactivity vs elapsedrecordtime; noACCinput',timing='existing nominal pairing; no lag fitting',test_and_28reserve='not opened',limitations='Activity association does not prove identity or full audio recoverability. Band association is descriptive, not causal bone-conduction proof.'),train_oof=record_macro([metrics(r['y'],p,cut) for r,p in zip(train,oof)]),train_oof_time_baseline=record_macro([metrics(r['y'],p,tcut) for r,p in zip(train,time_oof)]),probe_threshold=cut,time_threshold=tcut,train_selected_single_feature=selected,features=features,validation={})
probs={}
for r in val:
 p=predict(model,r);prior=time_profile(train,r);probs[r['id']]=p
 own=metrics(r['y'],p,cut);shift=metrics(np.roll(r['y'],50),p,cut);wrong=[];ownmatched=[]
 for other in val:
  if other['id']==r['id']:continue
  # Match by elapsedtime; use same available span and prediction samples for both labels.
  lo=max(r['t'][0],other['t'][0]);hi=min(r['t'][-1],other['t'][-1]);ix=(r['t']>=lo)&(r['t']<=hi)
  yy=np.interp(r['t'][ix],other['t'],other['y'])>=.5
  wa=auc(yy,p[ix]);oa=auc(r['y'][ix],p[ix])
  if wa is not None and oa is not None:wrong.append(wa);ownmatched.append(oa)
 interior=(r['t']>=.5)&(r['t']<=r['t'][-1]-.25)
 nonzero=r['audio']>-6.;span=(r['t']>=active_span[0])&(r['t']<=active_span[1])
 result['validation'][r['id']]=dict(acc_probe=own,time_only_baseline=metrics(r['y'],prior,tcut),always_active_baseline=metrics(r['y'],np.ones(len(p)),.5),shift500ms_probe=shift,wrong_record_auc=float(np.mean(wrong)),own_equalspan_auc=float(np.mean(ownmatched)),matched_minus_wrong_auc=float(np.mean(ownmatched)-np.mean(wrong)),edge500ms_sensitivity=metrics(r['y'][interior],p[interior],cut),nonzero_audio_only=metrics(r['y'][nonzero],p[nonzero],cut),nonzero_audio_time_baseline=metrics(r['y'][nonzero],prior[nonzero],tcut),train_active_span_only=metrics(r['y'][span],p[span],cut),train_active_span_time_baseline=metrics(r['y'][span],prior[span],tcut),digital_zero_fraction=float(np.mean(~nonzero)))
result['validation_macro']={k:record_macro([v[k] for v in result['validation'].values()]) for k in ['acc_probe','time_only_baseline','always_active_baseline','shift500ms_probe','edge500ms_sensitivity','nonzero_audio_only','nonzero_audio_time_baseline','train_active_span_only','train_active_span_time_baseline']}
result['validation_macro']['matched_minus_wrong_auc']=float(np.mean([v['matched_minus_wrong_auc'] for v in result['validation'].values()]))
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('TRAIN',result['train_oof'],'VAL',result['validation_macro'],'SELECTED',selected,features[selected],flush=True)

fig,axes=plt.subplots(4,1,figsize=(13,10))
j=names.index(selected)
for ax,r in zip(axes,val):
 p=probs[r['id']];ax.plot(r['t'],r['audio'],color='black',label='Microphone log RMS (left)');ax.axhline(threshold,color='gray',linestyle=':',label='TRAIN audio activity threshold')
 right=ax.twinx();right.fill_between(r['t'],0,1,where=r['y'].astype(bool),color='gray',alpha=.12,label='Audio-active reference');right.plot(r['t'],p,color='tab:blue',label='ACC-only activity probability (right)');right.plot(r['t'],time_profile(train,r),color='tab:orange',alpha=.7,label='Time-only baseline (right)');right.axhline(cut,color='tab:blue',linestyle=':',alpha=.5);right.set_ylim(0,1)
 ax.set_title(r['id']+' / audio-active fraction '+f"{r['y'].mean():.2f}",fontsize=10);ax.set_ylabel('log10 audio RMS');right.set_ylabel('Probability')
 if r is val[0]:ax.legend(fontsize=7,loc='lower left');right.legend(fontsize=7,loc='upper right')
axes[-1].set_xlabel('Elapsed recording time / seconds');fig.suptitle('Cross-modal activity correspondence: no requirement for matching harmonic structures\nWhole validation recordings, shared TRAIN-only rules; automatic audio-energy activity labels',fontsize=11);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'activity_correspondence.png',dpi=115)
print('DONE',OUT,flush=True)
