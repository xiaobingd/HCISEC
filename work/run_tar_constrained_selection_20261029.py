"""Development-only TAR90 selection and calibration; all models remain frozen."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_personalized_teacher_20261029 as prior
old=prior.old;ROOT=Path('/autodl-fs/data/mei_tar_constrained_selection_20261029');TARGET=.9;GRID=[0.,.25,.5,1.]

def threshold(genuine):
 g=np.sort(np.asarray(genuine));n=len(g);k=int(np.floor(n*(1-TARGET)+1e-9));t=float(g[min(k,n-1)])
 assert (g>=t).mean()+1e-12>=TARGET
 # Highest observed genuine score meeting the empirical acceptance constraint.
 assert not any((g>=x).mean()+1e-12>=TARGET for x in np.unique(g[g>t]))
 return t

def oof_quality(s,y,words,folds):
 decisions=np.zeros(len(y),dtype=bool);details=[]
 for tr,va in folds:
  assert not set(words[tr])&set(words[va]);t=threshold(s[tr][y[tr]]);decisions[va]=s[va]>=t
  details.append(dict(threshold=t,tar=float(decisions[va][y[va]].mean()),far=float(decisions[va][~y[va]].mean())))
 tar=float(decisions[y].mean());far=float(decisions[~y].mean());auc=float(old.sel.auc(y,s))
 return dict(tar=tar,far=far,auc=auc,hter=(1-tar+far)/2,shortfall=max(0.,TARGET-tar),folds=details)

def rank(row):
 q=row['quality'];return (round(q['shortfall'],12),q['far'],q['hter'],-q['auc'],row['raw_weight']+row['teacher_weight'],row['teacher_key'],row['raw_weight'],row['teacher_weight'])

def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);p=old.protocol(u);src=prior.ROOT/('U%d'%u);auth=prior.AUTH80/('U%d'%u)
 y=np.concatenate([p.labels[r] for r in ['auth_train','model_validation']])==u-1;words=np.concatenate([prior.first.words(p,r) for r in ['auth_train','model_validation']]);folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 bo=np.load(str(auth/'oof_private.npz'));to=np.load(str(src/'oof_private.npz'));candidates=[]
 for key in ['none']+prior.CONFIGS:
  for rw in GRID:
   for tw in ([0.] if key=='none' else GRID[1:]):
    s=bo['base']+rw*bo['raw']+(tw*to[key] if key!='none' else 0.)
    candidates.append(dict(teacher_key=key,raw_weight=rw,teacher_weight=tw,quality=oof_quality(s,y,words,folds)))
 chosen=min(candidates,key=rank);assert len(candidates)==40
 cached=old.sel.joblib.load(str(auth/'models_private.joblib'));dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));teachers={};models={}
 oldlock=json.loads((src/'locked_before_final.json').read_text());authlock=json.loads((auth/'locked_before_queries.json').read_text())
 def standardized(m,x):return (old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std']
 def get_teacher(key):
  if key not in teachers:
   ck=torch.load(str(src/(key+'_teacher_private.pt')),map_location=prior.D,weights_only=False);t=old.base.Teacher().to(prior.D);t.load_state_dict(ck['state']);t.eval();teachers[key]=t;models[key]=old.sel.joblib.load(str(src/(key+'_rbf_private.joblib')))
  return teachers[key]
 def cal_score(spec,role):
  s=standardized(cached['base'],dev[role+'__base'])
  if spec['raw_weight']:s+=spec['raw_weight']*standardized(cached['raw'],dev[role+'__raw'])
  if spec['teacher_weight']:s+=spec['teacher_weight']*standardized(models.setdefault(spec['teacher_key'],old.sel.joblib.load(str(src/(spec['teacher_key']+'_rbf_private.joblib')))),prior.embed(get_teacher(spec['teacher_key']),p.arrays[role]))
  return s
 bw=dict(teacher_key='none',raw_weight=0.,teacher_weight=0.)
 cw=oldlock['routes']['control']['weights'];cs=dict(teacher_key='control',raw_weight=cw.get('raw',0.),teacher_weight=cw.get('teacher',0.))
 specs={'baseline_original':dict(bw,threshold=authlock['thresholds']['baseline_rbf']),'control_original':dict(cs,threshold=oldlock['routes']['control']['threshold'])}
 for name,spec in [('baseline_tar90',bw),('control_tar90',cs),('selected_tar90',{k:chosen[k] for k in ['teacher_key','raw_weight','teacher_weight']})]:
  g=cal_score(spec,'target_dev_pool');n=cal_score(spec,'impostor_calibration');t=threshold(g)
  specs[name]=dict(spec,threshold=t,calibration_tar=float((g>=t).mean()),calibration_far=float((n>=t).mean()))
 old.save(folder/'locked_before_final.json',dict(target=u,target_tar=TARGET,candidates=candidates,chosen=chosen,routes=specs,final_seen=False))
 # Reuse locked scores only after development decisions have been persisted.
 bf=np.load(str(auth/'scores_private.npz'));rows={};dump={}
 for name,spec in specs.items():
  scores={}
  for role in ['final_genuine','final_impostor']:
   s=bf['baseline_rbf__'+role].copy()
   if spec['raw_weight']:
    raw=p.arrays[role];stats=np.concatenate([raw.mean(2),raw.std(2),raw.mean(1),raw.std(1)],1);s+=spec['raw_weight']*standardized(cached['raw'],stats)
   if spec['teacher_weight']:s+=spec['teacher_weight']*standardized(models[spec['teacher_key']],prior.embed(get_teacher(spec['teacher_key']),p.arrays[role]))
   scores[role]=s;dump[name+'__'+role]=s
  r=old.sel.evaluate_route(scores,p,u,spec['threshold']);r.update(spec);g=scores['final_genuine'];n=scores['final_impostor'];t=spec['threshold']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  rows[name]=r
 oldresult=json.loads((src/'results_public.json').read_text())['routes']['control']
 for k in ['tar','far','auc']:assert abs(rows['control_original'][k]-oldresult[k])<1e-12
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor']);out=dict(target=u,routes=rows,chosen=chosen,old_control_replay_verified=True)
 old.save(folder/'results_public.json',out);print('COMPLETE',u,json.dumps(out),flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);old.records();old.save(ROOT/'protocol_public.json',dict(targets=[4,5],target_calibration_tar=TARGET,grid=GRID,candidate_count=40,training='no retraining; frozen prior upstream teachers and auth80 RBF models',selection='reused training OOF:group5fold heldout decisions use other4fold genuine quantile90 threshold;prefer reaches pooled90TAR then lowestFAR;else smallestTARshortfall thenFAR;tiesHTER/AUC/smallerweights',calibration='highest observed genuine threshold with empirical TAR>=90;20 genuine only;negative scores diagnostic,not threshold tuning',routes=['old baseline','old control teacher','baseline new threshold','control new threshold','OOF-selected fusion new threshold'],all_eight_attackers=True,limitations=['historical final reused exploratory known identities','40candidate search on reused OOF,not fresh nested model-selection estimate','OOF per-sample predictions held out but source models overlap meta-validation folds;selection can overfit','cal20 gives coarse quantile and no populationTAR guarantee','branch RBF hyperparameters still selected by original AUC rule','TAR priority may substantially increaseFAR;do not call security improvement','fixed prior teachers;this changes selection/calibration,not teacher training']))
 rows={str(u):target(u) for u in [4,5]};old.save(ROOT/'results_public.json',rows);old.save(ROOT/'COMPLETE.json',dict(status='PASS'))
if __name__=='__main__':main()

