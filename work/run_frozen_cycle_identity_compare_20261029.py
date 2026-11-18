"""Stage 1: matched frozen raw/cycle/772/fixed-concatenation authentication."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_crosscontent_adapter_20261029 as trial
old=trial.old
ROOT=Path('/autodl-fs/data/mei_frozen_cycle_identity_compare_20261029')
ROLES=['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']

@torch.no_grad()
def encode(m,x):
 cur=x;shallow=deep=None
 for i,layer in enumerate(m.g_acc2audio.model):
  cur=layer(cur)
  if i==6:shallow=cur
  if i==13:deep=cur;break
 return torch.cat([m.shallow_agg(shallow),m.bottleneck_agg(deep)],1)

@torch.no_grad()
def extract(m,raw):
 parts={k:[] for k in ['raw640','cycle640','base772','concat1280']}
 for i in range(0,len(raw),8):
  x=old.te.tensor(raw[i:i+8]);cycle=m.g_audio2acc(m.g_acc2audio(x))
  a=encode(m,x);b=encode(m,cycle)
  values=dict(raw640=a,cycle640=b,base772=old.base.fusion_inputs(m,x),concat1280=torch.cat([a,b],1))
  for k,v in values.items():
   assert torch.isfinite(v).all();parts[k].append(v.cpu().numpy())
 return {k:np.concatenate(v) for k,v in parts.items()}

def words(p,role):
 return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])

def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True)
 p=old.protocol(u);m=old.load_gan(u)
 assert all(not q.requires_grad for q in m.parameters())
 state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 features={r:extract(m,p.arrays[r]) for r in ROLES[:4]}
 np.savez_compressed(str(folder/'development_private.npz'),**{r+'__'+k:v for r,a in features.items() for k,v in a.items()})
 labels=p.labels['auth_train'];y=labels==u-1;vy=p.labels['model_validation']==u-1
 models={};locks={}
 for key in features['auth_train']:
  best=None
  for c in [1.,10.,100.]:
   for gamma in ['scale',.1]:
    model=old.sel.rbf.fit([features['auth_train'][key]],y,c,gamma)
    score=old.sel.rbf.score(model,[features['model_validation'][key]])
    q=trial.quality(score,vy,words(p,'model_validation'));rank=(-q[0],-q[1],c,0 if gamma==.1 else 1)
    if best is None or rank<best[0]:best=(rank,model,c,gamma,q)
  model=best[1];models[key]=model
  cg=old.sel.rbf.score(model,[features['target_dev_pool'][key]]);cn=old.sel.rbf.score(model,[features['impostor_calibration'][key]])
  locks[key]=dict(C=best[2],gamma=best[3],validation_sameword_auc=best[4][0],validation_auc=best[4][1],threshold=float(old.base.original.choose_threshold(cg,cn)))
  old.sel.joblib.dump(model,str(folder/(key+'_private.joblib')))
 old.save(folder/'locked_before_final.json',dict(target=u,routes=locks,final_seen=False))
 for r in ROLES[4:]:features[r]=extract(m,p.arrays[r])
 rows={};dump={}
 for key,spec in locks.items():
  scores={r:old.sel.rbf.score(models[key],[features[r][key]]) for r in ROLES[4:]}
  result=old.sel.evaluate_route(scores,p,u,spec['threshold']);result.update(spec)
  g=scores['final_genuine'];n=scores['final_impostor'];t=spec['threshold']
  assert abs(float(((g[:,None]>n)+.5*(g[:,None]==n)).mean())-result['auc'])<1e-12
  assert abs(float((g>=t).mean())-result['tar'])<1e-12 and abs(float((n>=t).mean())-result['far'])<1e-12
  assert len(result['attack_by_user'])==8
  wg=words(p,'final_genuine');wn=words(p,'final_impostor')
  result['per_word']={str(w):old.sel.exp.metrics(g[wg==w],n[wn==w],t) for w in sorted(np.unique(wg))}
  result['same_word_mean_auc']=float(np.mean([v['auc'] for v in result['per_word'].values()]))
  rows[key]=result
  for r,v in scores.items():dump[key+'__'+r]=v
 assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in m.state_dict().items())
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 old.save(folder/'results_public.json',dict(target=u,routes=rows,frozen_state_verified=True,score_replay_verified=True))
 print('TARGET_COMPLETE',u,json.dumps(rows),flush=True)
 return rows

def main():
 ROOT.mkdir(exist_ok=True);old.records()
 old.save(ROOT/'protocol_public.json',dict(targets=[1,4,5],stage=1,features=['raw640','cycle640','base772','concat1280'],encoding='same original frozen personal G: shallow128 plus bottleneck mean/std512',authentication='matched binary RBF C1/10/100 gamma scale/.1; validation sameword meanAUC then pooledAUC; independent calibration threshold',split='existing strict content auth50/val30/cal20/final47-50 per identity; unchanged eight attackers',fusion='fixed concatenation, no learned gating',limitations=['historical final set previously observed; exploratory known-user closed set','word mapping conditional on10/20period and missing-tail assumption','shared frozen personal encoder may itself favor owner; this is not an independent identity teacher','640 raw and cycle are representations of the same encoder,not raw waveform direct classifier'],source_sha256=old.sha(Path(__file__))))
 results={}
 for u in [1,4,5]:
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(results)))
  results[str(u)]=target(u);old.save(ROOT/'results_public.json',results)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(results)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

