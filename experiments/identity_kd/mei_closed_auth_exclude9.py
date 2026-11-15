"""Restricted eight-user authentication; upstream nine-user models stay frozen."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.svm import SVC
from run_mei9_shape_v4 import Student,encode
from run_rkd01_vibravox import norm,thresholds,metrics
torch.set_num_threads(2)
p=Path('/autodl-fs/data/mei9_closed_rerun_20261110');out=Path('/autodl-fs/data/mei8_closed_auth_20261115');out.mkdir(exist_ok=True)
d=dict(np.load('/autodl-fs/data/mei9_controlled_identity_20261109/common_frontend_private.npz'));data=np.load(p/'prepared_private.npz');original={k:np.array(v) for k,v in json.loads((p/'split_private.json').read_text()).items()};ix={k:v[d['user'][v]!=9] for k,v in original.items()};allix=np.unique(np.r_[original['register'],original['calA'],original['calB'],original['sameword'],original['crossword']]);assert all(9 not in set(d['user'][ix[k]]) for k in ix)
net=Student();net.load_state_dict(torch.load(p/'student_private.pt',map_location='cpu',weights_only=False));base=np.c_[data['shape'],encode(net,data['x'])]/np.sqrt(2);raw=data['stats'];sc=joblib.load(p/'gan_scalers_private.joblib');cache={};branches=['input256','cycle256','cycle_stats720']
for u in range(1,9):
 cache[u]={}
 for step in [25,100]:
  q=np.load(p/f'owner{u}'/f'features{step}_{len(allix)}_private.npz');cache[u][step]={}
  for k in branches:
   v=np.zeros((len(base),q[k].shape[1]),dtype='float32');v[allix]=norm(sc[k].transform(q[k]));cache[u][step][k]=v
def view(u,c):return base if c['step']==0 else np.c_[base,c['alpha']*cache[u][c['step']][c['branch']]]/np.sqrt(1+c['alpha']**2)
def fit(v,u):return SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix['register']],d['user'][ix['register']]==u)
choices=[dict(step=0,branch='none',alpha=0.)]+[dict(step=s,branch=k,alpha=a) for s in [25,100] for k in branches for a in [.25,.5,1.]]
for c in choices:
 vv=[]
 for u in range(1,9):
  v=view(u,c);model=fit(v,u);a=ix['calA'];b=ix['calB'];tau=thresholds(model.decision_function(v[a]),d['user'][a]==u)['Cal_EER'];vv.append(metrics(model.decision_function(v[b]),d['user'][b]==u,tau))
 c.update(hter=float(np.mean([v['hter'] for v in vv])),auc=float(np.mean([v['auc'] for v in vv])))
best=min(choices,key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));(out/'selection_public.json').write_text(json.dumps(dict(selected=best,choices=choices),indent=2));print('CAL_LOCKED',best,flush=True);rows=[];private={}
for u in range(1,9):
 for name,v in dict(raw_statistics=raw,audio_baseline=base,selected_generator=view(u,best)).items():
  model=fit(v,u);cal=np.r_[ix['calA'],ix['calB']];ts=thresholds(model.decision_function(v[cal]),d['user'][cal]==u);private[f'{u}_{name}']=dict(model=model,thresholds=ts)
  for cond in ['sameword','crossword']:
   j=ix[cond];s=model.decision_function(v[j]);y=d['user'][j]==u
   for point,tau in ts.items():rows.append(dict(user=f'U{u}',method=name,condition=cond,point=point,**metrics(s,y,tau)))
 print('COMPLETE',u,flush=True)
joblib.dump(private,out/'heads_thresholds_private.joblib');(out/'per_user_public.json').write_text(json.dumps(rows,indent=2));summary=[]
for name in ['raw_statistics','audio_baseline','selected_generator']:
 for cond in ['sameword','crossword']:
  for pt in ['Cal_EER','Cal_FAR5']:
   r=[v for v in rows if v['method']==name and v['condition']==cond and v['point']==pt];assert len(r)==8;summary.append(dict(method=name,condition=cond,point=pt,**{k:float(np.mean([v[k] for v in r])) for k in ['tar','far','hter','auc']}))
audit=dict(state='PASS',authentication_users=8,excluded_user='U9',registration=96,calibration=64,sameword_query=128,crossword_query=192,known_attackers_each=7,all_authentication_roles_exclude_U9=True,upstream_models_still_trained_with_all9=True,query_used_for_selection=False,scope='restricted user cohort;not unseen-user experiment;nominal word labels')
(out/'verification_public.json').write_text(json.dumps(audit,indent=2));(out/'results_public.json').write_text(json.dumps(dict(state='complete',summary=summary,selected=best,audit=audit),indent=2));print(json.dumps(summary),flush=True)
