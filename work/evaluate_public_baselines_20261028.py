"""Frozen full772 baseline and matched raw/cycle controls, unit-isolated readouts."""
import sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve
ROOT=Path('/autodl-fs/data/public_baseline_compare_20261028');sys.path.insert(0,str(ROOT/'source'))
from model_drop import GeneratorAccToAudio,GeneratorAudioToAcc
DEVICE='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)

def threshold(g,n):
 v=np.unique(np.r_[g,n]);c=np.r_[v[0]-1e-6,(v[:-1]+v[1:])/2,v[-1]+1e-6]
 return float(min(c,key=lambda t:(abs(np.mean(n>=t)-np.mean(g<t)),(np.mean(n>=t)+np.mean(g<t))/2)))
def fit(x,y,c,gamma):
 s=StandardScaler().fit(x);model=SVC(C=c,gamma=gamma,class_weight='balanced').fit(s.transform(x)/np.sqrt(x.shape[1]),y);return s,model
def score(pair,x):return pair[1].decision_function(pair[0].transform(x)/np.sqrt(x.shape[1]))
def metrics(g,n,t):
 far=float(np.mean(n>=t));frr=float(np.mean(g<t));y=np.r_[np.ones(len(g)),np.zeros(len(n))];s=np.r_[g,n];fpr,tpr,_=roc_curve(y,s)
 return {'genuine_n':len(g),'attacker_n':len(n),'genuine_accepted':int(np.sum(g>=t)),'attacker_accepted':int(np.sum(n>=t)),'tar':1-frr,'far':far,'frr':frr,'hter':(far+frr)/2,'auc':float(roc_auc_score(y,s)),'tar_at_far5_test_roc':float(tpr[fpr<=.05].max())}
def pool(x):return torch.cat([x.mean((2,3)),x.flatten(2).std(2)],1)
@torch.no_grad()
def extract(g,f,x):
 out={k:[] for k in ['raw320','input_encoder640','cycle_encoder640','full772']};gen=[];cyc=[]
 for i in range(0,len(x),16):
  a=torch.tensor(x[i:i+16,None],device=DEVICE);m=g(a);c=f(m);d=a-c;sh=g.model[:7](a).mean((2,3));deep=pool(g.model[:14](a));flat=d.flatten(1)
  val={'raw320':torch.cat([a[:,0].mean(2),a[:,0].std(2),a[:,0].mean(1),a[:,0].std(1)],1),'input_encoder640':torch.cat([sh,deep],1),
       'cycle_encoder640':torch.cat([g.model[:7](c).mean((2,3)),pool(g.model[:14](c))],1),
       'full772':torch.cat([sh,deep,g.model[:7](d).mean((2,3)),torch.stack([flat.mean(1),flat.std(1),flat.abs().max(1).values,flat.square().mean(1).sqrt()],1)],1)}
  for k,v in val.items():out[k].append(v.cpu().numpy())
  gen.append(m[:,0].cpu().numpy());cyc.append(c[:,0].cpu().numpy())
 return {k:np.concatenate(v) for k,v in out.items()},np.concatenate(gen),np.concatenate(cyc)
def corr(a,b):
 a=a.reshape(len(a),-1);b=b.reshape(len(b),-1);a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
 return float(np.mean(np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)))

def main():
 results={};private={};partial='--partial' in sys.argv
 for dataset in ['esmb','vibravox']:
  data=ROOT/dataset;p=json.loads((data/'protocol_private.json').read_text());d=dict(np.load(str(data/'inputs.npz')));dataset_rows={}
  for owner in p['owners']:
   folder=data/owner;checkpoint=folder/'models/checkpoint_epoch_200.pth'
   if partial and not (checkpoint.exists() and ('MODEL_COMPLETE '+dataset+' '+owner) in (folder/'training.log').read_text(errors='replace')):continue
   assert checkpoint.exists()
   c=torch.load(str(checkpoint),map_location=DEVICE,weights_only=False);g=GeneratorAccToAudio(input_dim=1,hidden_dim=64,output_dim=1).to(DEVICE);f=GeneratorAudioToAcc(input_dim=1,hidden_dim=64,output_dim=1).to(DEVICE)
   g.load_state_dict(c['G_acc2audio_state_dict']);f.load_state_dict(c['G_audio2acc_state_dict']);g.eval();f.eval();g.requires_grad_(False);f.requires_grad_(False)
   checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();input_sha=hashlib.sha256((data/'inputs.npz').read_bytes()).hexdigest();cachepath=folder/'frozen_features_private.npz'
   if cachepath.exists():
    cache=dict(np.load(str(cachepath)));assert str(cache['checkpoint_sha'])==checkpoint_sha and str(cache['input_sha'])==input_sha
    generated=cache.pop('generated');cycle=cache.pop('cycle');cache.pop('checkpoint_sha');cache.pop('input_sha');bank=cache
   else:
    bank,generated,cycle=extract(g,f,d['acc']);np.savez_compressed(str(cachepath),generated=generated,cycle=cycle,checkpoint_sha=np.array(checkpoint_sha),input_sha=np.array(input_sha),**bank)
   units=np.unique(d['unit_id']);speaker={int(v):str(d['speaker_id'][d['unit_id']==v][0]) for v in units}
   agg={key:{int(u):val[d['unit_id']==u].mean(0) for u in units} for key,val in bank.items()}
   known=next(v for v in p['owners'] if v!=owner);fitids=p['roles'][owner]['auth_train']+p['roles'][known]['auth_train'];y=np.array([speaker[v]==owner for v in fitids]);rows={};locks={}
   for key in bank:
    get=lambda ids:np.stack([agg[key][int(v)] for v in ids]);x=get(fitids);candidates=[];folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))
    for C in [1.,10.,100.]:
     for gamma in ['scale',.1]:
      auc=[]
      for tr,va in folds:auc.append(float(roc_auc_score(y[va],score(fit(x[tr],y[tr],C,gamma),x[va]))))
      candidates.append({'C':C,'gamma':gamma,'cv_auc':float(np.mean(auc))})
    config=min(candidates,key=lambda z:(-z['cv_auc'],z['C'],str(z['gamma'])));model=fit(x,y,config['C'],config['gamma']);cg=score(model,get(p['roles'][owner]['calibration']));cn=score(model,get(p['roles'][known]['calibration']));t=threshold(cg,cn)
    locks[key]={'config':config,'threshold':t,'fit_units':fitids,'calibration_positive_units':p['roles'][owner]['calibration'],'calibration_negative_units':p['roles'][known]['calibration']}
    (folder/('auth_'+key+'_locked_private.json')).write_text(json.dumps(locks[key],indent=2));joblib.dump({'scaler':model[0],'svm':model[1],'lock':locks[key]},str(folder/('auth_'+key+'_model.joblib')))
    # Only score final after this route's CV/model/threshold is fixed.
    sg=score(model,get(p['roles'][owner]['final']));ss={v:score(model,get(p['roles'][v]['final'])) for v in [known]+p['unknown']};unseen=np.concatenate([ss[v] for v in p['unknown']])
    rows[key]={'known':metrics(sg,ss[known],t),'unknown':metrics(sg,unseen,t),'per_attacker':{v:{'n':len(s),'accepted':int(np.sum(s>=t)),'mean_score':float(s.mean())} for v,s in ss.items()},'config':config,'threshold':t}
    private[dataset+'_'+owner+'_'+key+'_g']=sg
    for v,s in ss.items():private[dataset+'_'+owner+'_'+key+'_'+v]=s
   genfit=np.isin(d['unit_id'],p['roles'][owner]['gan_train']);genval=np.isin(d['unit_id'],p['roles'][owner]['gan_val']);test=np.isin(d['unit_id'],p['roles'][owner]['final'])
   quality={}
   for name,mask in [('upstream_train',genfit),('upstream_validation',genval),('auth_final',test)]:
    quality[name]={'windows':int(mask.sum()),'audio_mae':float(np.mean(np.abs(generated[mask]-d['audio'][mask]))),'audio_corr':corr(generated[mask],d['audio'][mask]),'audio_dynamic_corr':corr(generated[mask]-generated[mask].mean(2,keepdims=True),d['audio'][mask]-d['audio'][mask].mean(2,keepdims=True)),'audio_envelope_corr':corr(generated[mask].mean(1),d['audio'][mask].mean(1)),'cycle_acc_mae':float(np.mean(np.abs(cycle[mask]-d['acc'][mask]))),'cycle_acc_corr':corr(cycle[mask],d['acc'][mask])}
   dataset_rows[owner]={'generation':quality,'routes':rows,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
   (folder/'auth_locked_private.json').write_text(json.dumps(locks,indent=2));print('OWNER_EVALUATED',dataset,owner,rows['full772'],flush=True)
   # Plain scientific plot, same color limits for real and generated modality.
   import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
   ix=np.flatnonzero(test)[:3];fig,axs=plt.subplots(4,3,figsize=(10,9))
   for j,i in enumerate(ix):
    for a,(arr,title) in enumerate([(d['acc'],'Real BC/ACC'),(d['audio'],'Real Audio Mel'),(generated,'Generated Audio Mel'),(cycle,'Cycle BC/ACC')]):
     axs[a,j].imshow(arr[i],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');axs[a,j].set_title(title);axs[a,j].set_xticks([]);axs[a,j].set_yticks([])
   fig.suptitle(dataset+' '+owner+' | isolated auth-final windows | fixed epoch200');fig.tight_layout();fig.savefig(str(folder/'auth_final_generation.png'),dpi=140);plt.close(fig)
  results[dataset]=dataset_rows;(ROOT/('results_partial.json' if partial else 'results_public.json')).write_text(json.dumps(results,indent=2))
 if partial:
  np.savez_compressed(str(ROOT/'partial_private_scores.npz'),**private);print('PARTIAL_BASELINES_EVALUATED',flush=True);return
 np.savez_compressed(str(ROOT/'private_scores.npz'),**private)
 subprocess.run([sys.executable,'/root/verify_public_baselines_20261028.py'],check=True)
 (ROOT/'COMPLETE').write_text('all four models, authentication and independent score verification complete\n');print('ALL_BASELINES_COMPLETE',flush=True)

if __name__=='__main__':main()
