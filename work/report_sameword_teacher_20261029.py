"""Independent count, word/repetition balance, model-score and same-word AUC verification."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0,'/root')
import run_sameword_teacher_baseline_20261029 as trial
old=trial.old;ROOT=trial.ROOT
M=json.loads((ROOT/'manifest_private.json').read_text());rows={}
for v,source in M['sources'].items():
 roles=source['roles'];assert [len(roles[k]) for k in ['auth_train','calibration','test']]==[75,30,30]
 flat=sum(roles.values(),[]);assert len(flat)==len(set(flat))==135
 upstream=old.M['sources'][v]['roles'];assert not set(flat)&set(upstream['gan_fit']+upstream['gan_val'])
 for role,rr in M['repeat_roles'].items():
  assert {(source['groups'][i],source['repeats'][i]) for i in roles[role]}=={(w,r) for w in M['auth_groups'] for r in rr}
 assert roles==M['sources']['1']['roles']
for u in [1,4,5]:
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text());locked=json.loads((folder/'locked_before_final.json').read_text())
 models=old.sel.joblib.load(str(folder/'models_private.joblib'));saved=np.load(str(folder/'scores_private.npz'));oof=np.load(str(folder/'oof_private.npz'))
 ids=[str(v)+':'+str(j) for v in old.USERS for j in M['sources'][str(v)]['roles']['test']]
 labels=np.asarray([int(s.split(':')[0])-1 for s in ids]);words=np.asarray([M['sources'][s.split(':')[0]]['groups'][int(s.split(':')[1])] for s in ids])
 assert np.array_equal(labels,saved['labels']) and np.array_equal(words,saved['words'])
 p=old.protocol(u);dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(trial.FINAL/('U%d'%u)/'final_features_private.npz'))
 allids=[];parts={k:[] for k in ['base','old_acc','old_audio','identity_acc','identity_audio','robust_acc','robust_audio']}
 for role in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  allids+=p.sample_ids[role];source=final if role.startswith('final_') else dev
  for k in parts:parts[k].append(source[role+'__'+k])
 index={s:i for i,s in enumerate(allids)};matrix={k:np.concatenate(v)[[index[s] for s in ids]] for k,v in parts.items()}
 train_ids=[str(v)+':'+str(j) for v in old.USERS for j in M['sources'][str(v)]['roles']['auth_train']]
 trainlabels=np.asarray([int(s.split(':')[0])-1 for s in train_ids]);repeats=np.asarray([M['sources'][s.split(':')[0]]['repeats'][int(s.split(':')[1])] for s in train_ids]);y=trainlabels==u-1
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,repeats))
 for route,r in a['routes'].items():
  if route=='train_selected_teacher':continue
  m=models[route]['model'];scalers,classifier=m;assert len(scalers)==1 and int(scalers[0].n_samples_seen_)==675
  if route=='baseline772':x=matrix['base']
  else:
   family=route.split('_teacher900')[0];x=np.concatenate([matrix['base'],matrix[family+'_acc'],matrix[family+'_audio']],axis=1)
  scores=classifier.decision_function(scalers[0].transform(x)/np.sqrt(x.shape[1]));assert np.allclose(scores,saved[route],atol=1e-12,rtol=1e-12)
  values=[float(roc_auc_score(y[va],oof[route][va])) for _,va in folds];assert np.allclose(values,locked['routes'][route]['fold_auc'],atol=1e-12,rtol=0)
  t=r['threshold'];g=scores[labels==u-1];n=scores[labels!=u-1]
  assert len(g)==r['genuine_n']==30 and len(n)==r['impostor_n']==240 and len(r['attack_by_user'])==8
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
  for v,b in r['attack_by_user'].items():
   score=scores[labels==int(v)-1];assert len(score)==b['n']==30 and int((score>=t).sum())==b['accepted']
  for w in M['auth_groups']:
   mask=words==w;positive=labels[mask]==u-1;score=scores[mask];rr=r['per_word'][M['word_vocabulary'][w]]
   assert positive.sum()==rr['genuine_n']==2 and (~positive).sum()==rr['impostor_n']==16
   assert abs(roc_auc_score(positive,score)-rr['auc'])<1e-12
   assert abs((score[positive]>=t).mean()-rr['tar'])<1e-12 and abs((score[~positive]>=t).mean()-rr['far'])<1e-12
  assert abs(np.mean([rr['auc'] for rr in r['per_word'].values()])-r['same_word_mean_auc'])<1e-12
  r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
 selected=min([k for k in locked['routes'] if k!='baseline772'],key=lambda k:(-np.mean(locked['routes'][k]['fold_auc']),k));assert selected==locked['teacher_selected']
 a['routes']['train_selected_teacher']=dict(a['routes'][selected],selected_family=selected);rows[str(u)]=a
macro={k:{m:float(np.mean([a['routes'][k][m] for a in rows.values()])) for m in ['tar','far','auc','hter','same_word_mean_auc']} for k in rows['1']['routes']}
out=dict(status='PASS',targets=3,results=rows,macro=macro,protocol={k:v for k,v in M.items() if k!='sources'},verification='Exact same word/repeat/row design across nine identities;upstream135-vs150 disjointness;train scaler counts;model and OOF scores;all eight attacks and per-word2-vs16 AUC/counts;teacher family chosen TRAIN only')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

