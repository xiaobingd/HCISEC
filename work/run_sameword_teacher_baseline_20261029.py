"""Content-matched repeated-recording baseline; U1/U4/U5, all eight attackers."""
import sys,json,os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_sameword_teacher_20261029')
FINAL=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
WORDS=['zero','one','two','three','four','five','six','seven','eight','nine','Start','Pause','Okay','Stop','Menu','Select','Cancel','Confirm','Exit','Help','Next','Back','Off','Grab','Down','Up','Zoom','Record','Play','Reset']
GROUPS=[5,6,7,8,9]+list(range(20,30))
REPEATS={'auth_train':list(range(5)),'calibration':[5,6],'test':[7,8]}
def save(p,x):old.save(p,x)
def manifest():
 old.records();sources={}
 for v in old.USERS:
  src=old.M['sources'][str(v)];g=np.asarray(src['content_groups']);i=np.arange(src['rows']);rep=np.where(i<100,i//10,(i-100)//20)
  roles={role:np.flatnonzero(np.isin(g,GROUPS)&np.isin(rep,rr)).tolist() for role,rr in REPEATS.items()}
  assert [len(roles[k]) for k in REPEATS]==[75,30,30]
  for role,rr in REPEATS.items():
   assert {(int(g[j]),int(rep[j])) for j in roles[role]}=={(w,r) for w in GROUPS for r in rr}
  flat=sum(roles.values(),[]);assert len(set(flat))==135 and not set(flat)&set(src['roles']['gan_fit']+src['roles']['gan_val'])
  sources[str(v)]=dict(roles=roles,groups=g.tolist(),repeats=rep.tolist(),source_sha256=src['sha256'])
 # Identical row positions, word and repeat design across all nine identities.
 assert all(sources[str(v)]['roles']==sources['1']['roles'] for v in old.USERS)
 return dict(targets=[1,4,5],users=old.USERS,word_vocabulary=WORDS,auth_groups=GROUPS,auth_words=[WORDS[k] for k in GROUPS],repeat_roles=REPEATS,sources=sources,upstream='Existing strict personal G/F200 and frozen personal teacher families; original fit120/val30, none of these135 authentication rows exposed',mapping_evidence='User confirmed ordered vocabulary plus previous Audio10/20period audit; not per-record independently transcribed',excluded='Repeat9 uniformly excluded for all identities to avoid incomplete-tail imbalance;no error-based filtering',seed=42)
def target(u,M):
 folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);p0=old.protocol(u)
 dev=dict(np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz')))
 cache=np.load(str(FINAL/('U%d'%u)/'final_features_private.npz'))
 for k in cache.files:dev[k]=cache[k]
 names=['base','old_acc','old_audio','identity_acc','identity_audio','robust_acc','robust_audio']
 allids=[];allfeatures={k:[] for k in names}
 for role in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  allids+=p0.sample_ids[role]
  for name in names:allfeatures[name].append(dev[role+'__'+name])
 assert len(allids)==len(set(allids));index={s:i for i,s in enumerate(allids)};allfeatures={k:np.concatenate(v) for k,v in allfeatures.items()}
 roleids={role:[str(v)+':'+str(j) for v in old.USERS for j in M['sources'][str(v)]['roles'][role]] for role in REPEATS}
 labels={role:np.asarray([int(s.split(':')[0])-1 for s in ids]) for role,ids in roleids.items()}
 words={role:np.asarray([M['sources'][s.split(':')[0]]['groups'][int(s.split(':')[1])] for s in ids]) for role,ids in roleids.items()}
 repeats={role:np.asarray([M['sources'][s.split(':')[0]]['repeats'][int(s.split(':')[1])] for s in ids]) for role,ids in roleids.items()}
 # Explicitly select fit/calibration features only; final matrix construction deferred until lock.
 features={role:{k:v[[index[s] for s in roleids[role]]] for k,v in allfeatures.items()} for role in ['auth_train','calibration']}
 def matrices(x):
  out={'baseline772':x['base']}
  for family in ['old','identity','robust']:out[family+'_teacher900']=np.concatenate([x['base'],x[family+'_acc'],x[family+'_audio']],axis=1)
  return out
 train=matrices(features['auth_train']);cal=matrices(features['calibration']);y=labels['auth_train']==u-1
 assert y.sum()==75 and (~y).sum()==600
 # Hold out one repetition across all words and identities: every fold preserves vocabulary.
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,repeats['auth_train']))
 for tr,va in folds:
  assert not set(repeats['auth_train'][tr])&set(repeats['auth_train'][va])
  assert set(words['auth_train'][tr])==set(words['auth_train'][va])==set(GROUPS)
 models={};oof={};thresholds={};history={}
 for route,x in train.items():
  oof[route],models[route]=old.sel.nested_branch(x,y,repeats['auth_train'],folds)
  m=models[route];ss=old.sel.rbf.score(m['model'],[cal[route]]);pos=labels['calibration']==u-1
  thresholds[route]=float(old.base.original.choose_threshold(ss[pos],ss[~pos]))
  history[route]=dict(fold_auc=[float(roc_auc_score(y[va],oof[route][va])) for _,va in folds],C=m['C'],gamma=m['gamma'])
 selected=min([k for k in train if k!='baseline772'],key=lambda k:(-np.mean(history[k]['fold_auc']),k))
 locked=dict(target=u,routes=history,thresholds=thresholds,teacher_selected=selected,selection='Train-repeat nested5outer/3inner meanAUC; selected teacher family only',new_final_features_selected=False)
 save(folder/'locked_before_final.json',locked);old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 # Historical final features already exist; this is not a new blind data collection.
 test=matrices({k:v[[index[s] for s in roleids['test']]] for k,v in allfeatures.items()});yy=labels['test']==u-1
 rows={};dump={}
 for route,x in test.items():
  score=old.sel.rbf.score(models[route]['model'],[x]);g=score[yy];n=score[~yy];t=thresholds[route]
  result=old.sel.exp.metrics(g,n,t);result.update(feature_dimensions=x.shape[1],threshold=t,attack_by_user={},per_word={})
  for v in old.USERS:
   if v==u:continue
   s=score[labels['test']==v-1];result['attack_by_user'][str(v)]=dict(n=len(s),accepted=int((s>=t).sum()))
  for w in GROUPS:
   mask=words['test']==w;wp=yy[mask];ws=score[mask]
   assert wp.sum()==2 and (~wp).sum()==16
   result['per_word'][WORDS[w]]=old.sel.exp.metrics(ws[wp],ws[~wp],t)
  result['same_word_mean_auc']=float(np.mean([r['auc'] for r in result['per_word'].values()]))
  rows[route]=result;dump[route]=score
 rows['train_selected_teacher']=dict(rows[selected],selected_family=selected)
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,labels=labels['test'],words=words['test'],repeats=repeats['test'])
 save(folder/'results_public.json',dict(target=u,routes=rows,counts=dict(fit_positive=75,fit_negative=600,cal_positive=30,cal_negative=240,test_positive=30,test_negative=240),mapping_scope=M['mapping_evidence'],scope='All eight known attackers retained;identical15words,repeat-design and counts per identity;conditional on confirmed vocabulary+periodic row mapping;not per-file semantic verification'))
 print('TARGET_DONE',u,selected,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 M=manifest();save(ROOT/'manifest_private.json',M)
 public={k:v for k,v in M.items() if k!='sources'};public['counts_per_user']={role:len(M['sources']['1']['roles'][role]) for role in REPEATS};public['source_sha256']={k:v['source_sha256'] for k,v in M['sources'].items()};public['source_sha256_code']=old.sha(Path(__file__))
 save(ROOT/'protocol_public.json',public);rows={}
 for u in M['targets']:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));target(u,M)
  rows[str(u)]=json.loads((ROOT/('U%d'%u)/'results_public.json').read_text());save(ROOT/'results_public.json',rows)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':main()

