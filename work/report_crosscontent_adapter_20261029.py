"""Replay adapter checkpoints/scores, loss definitions and seed-averaged evaluation."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_crosscontent_adapter_20261029 as trial
old=trial.old;ROOT=trial.ROOT
def loss_checks():
 torch.manual_seed(10);raw=torch.randn(8,8,requires_grad=True);z=F.normalize(raw,dim=1)
 labels=torch.tensor([0,0,0,0,1,1,1,1]);words=torch.tensor([0,0,1,2,0,0,1,2]);got=trial.contrast(z,labels,words,True)
 a=z.detach().numpy();sim=a@a.T/.07;values=[]
 for i in range(8):
  pos=[j for j in range(8) if labels[i]==labels[j] and words[i]!=words[j]]
  den=sum(float(np.exp(sim[i,j]))*(2 if labels[i]!=labels[j] and words[i]==words[j] else 1) for j in range(8) if labels[i]!=labels[j] or j in pos)
  values.append(-np.mean([sim[i,j]-np.log(den) for j in pos]))
 assert abs(float(got)-np.mean(values))<2e-5;got.backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
 ll=np.repeat(np.arange(9),4);ww=np.tile([0,0,1,1],9);zz=F.normalize(torch.randn(36,8),dim=1)
 assert torch.isfinite(trial.episode(zz,ll,ww,np.random.RandomState(42)))
 return 'PASS manual weighted contrast denominator/positive mask, finite nonzero gradient and different-word episodic sampling'
def check_metrics(g,n,t,r):
 assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
checks=loss_checks();rows={}
for u in [1,4,5]:
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text());locked=json.loads((folder/'locked_before_final.json').read_text());p=old.protocol(u)
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(trial.FINAL/('U%d'%u)/'final_features_private.npz'));saved=np.load(str(folder/'scores_private.npz'))
 scaler=old.sel.joblib.load(str(folder/'scaler_private.joblib'));assert int(scaler.n_samples_seen_)==450
 labels=p.labels['auth_train'];x=trial.tensor(scaler.transform(dev['auth_train__base']))
 assert np.array_equal(saved['attack_labels'],p.labels['final_impostor'])
 def wordids(role):return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
 wg=wordids('final_genuine');wn=wordids('final_impostor')
 for key,r in a['routes'].items():
  spec=locked['routes'][key]
  if key=='rbf_matched':model=old.sel.joblib.load(str(folder/'rbf_private.joblib'))
  elif key!='raw772_cosine':
   hist=locked['history'][key];selected=min(hist,key=lambda t:(-t['validation_sameword_auc'],-t['validation_auc'],t['validation_multiclass_ce'],t['step']))
   assert selected['step']==spec['selected_step']
   model=trial.Adapter().to(trial.DEVICE);weights=torch.load(str(folder/(key+'_private.pt')),map_location=trial.DEVICE,weights_only=False)
   model.load_state_dict(weights['state']);model.eval();assert weights['metadata']['selected_step']==spec['selected_step']
  def score(raw):
   if key=='rbf_matched':return old.sel.rbf.score(model,[raw])
   with torch.no_grad():
    if key=='raw772_cosine':z=F.normalize(trial.tensor(scaler.transform(raw)),dim=1);cc=trial.centers(F.normalize(x,dim=1),labels)
    else:z=model(trial.tensor(scaler.transform(raw)));cc=trial.centers(model(x),labels)
    return (z@cc[u-1]).cpu().numpy()
  if key not in ['rbf_matched','raw772_cosine']:
   vs=score(dev['model_validation__base']);quality=trial.quality(vs,p.labels['model_validation']==u-1,wordids('model_validation'))
   assert abs(quality[0]-spec['validation_sameword_auc'])<1e-6 and abs(quality[1]-spec['validation_auc'])<1e-6
  cg=score(dev['target_dev_pool__base']);cn=score(dev['impostor_calibration__base']);threshold=old.base.original.choose_threshold(cg,cn)
  assert abs(threshold-r['threshold'])<2e-6
  for role in ['final_genuine','final_impostor']:assert np.allclose(score(final[role+'__base']),saved[key+'__'+role],atol=2e-6,rtol=1e-6)
  g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];t=r['threshold'];check_metrics(g,n,t,r)
  assert len(r['attack_by_user'])==8
  for v,b in r['attack_by_user'].items():
   s=n[saved['attack_labels']==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
  for w,rr in r['per_word'].items():check_metrics(g[wg==int(w)],n[wn==int(w)],t,rr)
  assert abs(np.mean([rr['auc'] for rr in r['per_word'].values()])-r['same_word_mean_auc'])<1e-12
 summary={}
 for mode in trial.MODES:
  summary[mode]={metric:dict(mean=float(np.mean([a['routes'][mode+'_seed%d'%s][metric] for s in trial.SEEDS])),std=float(np.std([a['routes'][mode+'_seed%d'%s][metric] for s in trial.SEEDS],ddof=1))) for metric in ['tar','far','auc','hter','same_word_mean_auc']}
 a['seed_summary']=summary;rows[str(u)]=a
macro={mode:{metric:float(np.mean([a['seed_summary'][mode][metric]['mean'] for a in rows.values()])) for metric in ['tar','far','auc','hter','same_word_mean_auc']} for mode in trial.MODES}
for route in ['rbf_matched','raw772_cosine']:macro[route]={metric:float(np.mean([a['routes'][route][metric] for a in rows.values()])) for metric in ['tar','far','auc','hter','same_word_mean_auc']}
out=dict(status='PASS',targets=3,routes_verified=sum(len(a['routes']) for a in rows.values()),results=rows,macro=macro,loss_checks=checks,verification='Checkpoint rank replay, train scaler count, saved weights re-embedding, validation scores, calibration thresholds, pairwise AUC/all counts/eight attack sources/per-word counts;3seeds averaged without final selection')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

