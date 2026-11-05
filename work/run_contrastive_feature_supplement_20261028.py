"""Retain all 772 baseline features; use contrastive embedding as optional add-on."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import joblib
sys.path.insert(0,'/root')
import run_contrastive_head_methods_20261028 as e
OUT=Path('/autodl-fs/data/contrastive_feature_supplement_20261028')
OriginalHead=e.Head
class Head(OriginalHead):
 def __init__(self,variant):
  super().__init__('compact64' if variant=='mlp64' else variant)
  if variant=='mlp64':self.identity=e.nn.Sequential(e.nn.Linear(772,128),e.nn.LayerNorm(128),e.nn.SiLU(),e.nn.Dropout(.1),e.nn.Linear(128,64))
e.Head=Head
VARIANTS=('compact64','semi_hard64','factorized64_16','mlp64');EPOCHS=(5,10,20,50);WEIGHTS=(.25,1.)

def fit(blocks,y,c,gamma):
 scalers=[StandardScaler().fit(x) for x,w in blocks]
 x=np.concatenate([sc.transform(x)*w/np.sqrt(x.shape[1]) for sc,(x,w) in zip(scalers,blocks)],1)
 model=SVC(C=c,gamma=gamma,class_weight='balanced').fit(x,y)
 return scalers,model

def score(fitted,blocks):
 scalers,model=fitted
 x=np.concatenate([sc.transform(x)*w/np.sqrt(x.shape[1]) for sc,(x,w) in zip(scalers,blocks)],1)
 return model.decision_function(x)

def main():
 OUT.mkdir(parents=True,exist_ok=True);pm=e.protocol_module();result={}
 for u in (1,4,5):
  print('TARGET',u,flush=True);p=pm.build_target_protocol(u,42,42)
  assert p.manifest==json.loads((e.FEATURE_ROOT/('U%d_split_manifest.json'%u)).read_text())
  pm.save_manifest(p,str(OUT/('U%d_split_manifest.json'%u)))
  archive=np.load(str(e.FEATURE_ROOT/('U%d_features.npz'%u)))
  xx={k:archive[k+'__base'] for k,v in p.arrays.items() if len(v)}
  ids=p.labels['auth_train'];y=ids==u-1
  words=np.asarray([int(str(x).split(':')[1])//10 for x in p.sample_ids['auth_train']])
  folds=list(e.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
  baseline=e.baseline_cv(xx['auth_train'],y,words,folds);c=baseline['C'];gamma=baseline['gamma']
  locked={'baseline':baseline}
  for variant in VARIANTS:
   cache=OUT/('U%d_%s_cv.json'%(u,variant))
   if cache.exists():locked[variant]=json.loads(cache.read_text());continue
   candidates={(epoch,alpha):{'epoch':epoch,'alpha':alpha,'by_seed':{str(seed):[] for seed in e.SEEDS}} for epoch in EPOCHS for alpha in WEIGHTS}
   for seed in e.SEEDS:
    for fold,(tr,va) in enumerate(folds):
     assert not set(words[tr])&set(words[va])
     scaler,model,states,_=e.fit_projection(xx['auth_train'][tr],ids[tr],words[tr],u,variant,seed+1000*fold,max(EPOCHS),EPOCHS)
     for epoch,state in states.items():
      model.load_state_dict(state);zt=e.embed(model,scaler,xx['auth_train'][tr]);zv=e.embed(model,scaler,xx['auth_train'][va])
      for alpha in WEIGHTS:
       fitted=fit([(xx['auth_train'][tr],1.),(zt,alpha)],y[tr],c,gamma)
       values=score(fitted,[(xx['auth_train'][va],1.),(zv,alpha)])
       candidates[epoch,alpha]['by_seed'][str(seed)].append(float(roc_auc_score(y[va],values)))
    print('CV_SEED_DONE',u,variant,seed,flush=True)
   rows=[]
   for row in candidates.values():
    row['mean_fold_auc']=float(np.mean([np.mean(v) for v in row['by_seed'].values()]));rows.append(row)
   best=min(rows,key=lambda r:(-r['mean_fold_auc'],r['alpha'],r['epoch']))
   locked[variant]={'selected':best,'candidates':rows};cache.write_text(json.dumps(locked[variant],indent=2))
   print('SUPPLEMENT_LOCKED',u,variant,best,flush=True)
  ranks={'baseline':baseline['mean_fold_auc']};ranks.update({v:locked[v]['selected']['mean_fold_auc'] for v in VARIANTS})
  selected=max(ranks,key=lambda k:(ranks[k],k=='baseline'))
  (OUT/('U%d_locked.json'%u)).write_text(json.dumps({'selected':selected,'choices':locked,'readout_C':c,'readout_gamma':gamma},indent=2))
  rows={};dumped={}
  for variant in ('baseline',)+VARIANTS:
   per_seed=[];seeds=(42,) if variant=='baseline' else e.SEEDS
   for seed in seeds:
    state=None
    if variant=='baseline':config=baseline;blocks={k:[(x,1.)] for k,x in xx.items()}
    else:
     config=locked[variant]['selected'];scaler,head,_,trace=e.fit_projection(xx['auth_train'],ids,words,u,variant,seed,config['epoch'])
     embeddings={k:e.embed(head,scaler,x) for k,x in xx.items()}
     blocks={k:[(x,1.),(embeddings[k],config['alpha'])] for k,x in xx.items()}
     state={'variant':variant,'state':{k:v.detach().cpu() for k,v in head.state_dict().items()},'scaler':scaler,'trace':trace}
    fitted=fit(blocks['auth_train'],y,c,gamma)
    cg=score(fitted,blocks['target_dev_pool']);cn=score(fitted,blocks['impostor_calibration'])
    t=float(e.exp.base.original.choose_threshold(cg,cn))
    joblib.dump({'projection':state,'classifier':fitted,'threshold':t,'config':config},str(OUT/('U%d_%s_seed%d.joblib'%(u,variant,seed))))
    sg=score(fitted,blocks['final_genuine']);sn=score(fitted,blocks['final_impostor']);row=e.metrics(sg,sn,t);row['seed']=seed;row['attack_by_user']={}
    for uid in np.unique(p.labels['final_impostor']):
     s=sn[p.labels['final_impostor']==uid];row['attack_by_user'][str(int(uid)+1)]={'n':len(s),'accepted':int(np.sum(s>=t))}
    per_seed.append(row);dumped[variant+'_genuine_'+str(seed)]=sg;dumped[variant+'_impostor_'+str(seed)]=sn
   summary={k:{'mean':float(np.mean([r[k] for r in per_seed])),'std':float(np.std([r[k] for r in per_seed]))} for k in ('tar','far','auc','hter','eer')}
   rows[variant]={'config':config,'per_seed':per_seed,'summary':summary};print('FINAL_DONE',u,variant,summary,flush=True)
  result[str(u)]={'selected_on_training_cv':selected,'variants':rows,'readout_C':c,'readout_gamma':gamma,'counts':p.manifest['counts'],
   'control':'All original 772 features retained; contrastive 64-D supplement; same baseline-selected SVM hyperparameters across branches',
   'status':'Exploratory existing-recording evaluation; no final-based selection'}
  np.savez_compressed(str(OUT/('U%d_final_scores.npz'%u)),**dumped)
  tmp=OUT/'results.tmp.json';tmp.write_text(json.dumps(result,indent=2));tmp.replace(OUT/'results.json')
  print('TARGET_DONE',u,'SELECTED',selected,flush=True)
 print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
