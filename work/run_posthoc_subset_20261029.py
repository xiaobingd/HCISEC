"""POST-HOC TEST-INFORMED restricted-cohort experiment; not independent validation."""
import sys,json,os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_posthoc_subset_20261029')
PRIOR=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
FIT=Path('/autodl-fs/data/mei_auth80_20261029')
LAMBDAS=[0.,.25,.5,1.,1.5,2.]
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def proto_fit(x,labels):
 s=StandardScaler().fit(x);z=norm(s.transform(x));uids=np.unique(labels);centers=norm(np.stack([z[labels==v].mean(0) for v in uids]));return s,uids,centers
def proto_pair(model,x,u):
 s,uids,c=model;sim=norm(s.transform(x))@c.T
 return sim[:,uids==u-1].ravel(),sim[:,uids!=u-1].max(1)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(remove_counts=[0,1,2],ranking='Highest prior baseline RBF FINAL FAR per attacker, ties UID; deliberately test-informed',fit_positive=80,fit_negative=[640,560,480],removed_from=['auth_fit','scaler_and_centers_fit','hyperparameter_selection','threshold_calibration'],final_evaluation='Primary excludes selected identities; full-eight scores retained as diagnostic; TEST-INFORMED selection',upstream_unchanged=True,scope='Restricted attacker cohort selected using prior final results; NOT independent final performance or open-set validation; upstream remains all nine users',preprocessing_unchanged=True,seed=42,limitations=['Exploratory reused final data','Single seed','Hard identity does not imply poor signal quality','Content mapping inferred10/20period and missing-tail assumption'],source_sha256=old.sha(Path(__file__))))
 allrows={}
 for u in old.ORDER:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(allrows)))
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);p=old.protocol(u)
  dev=dict(np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz')))
  labels=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')]);y=labels==u-1
  words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
  ids=sum([p.sample_ids[k] for k in ('auth_train','model_validation')],[])
  blocked={str(v)+':'+str(i) for v in old.USERS for k in ('gan_fit','gan_val','calibration','test') for i in old.M['sources'][str(v)]['roles'][k]}
  assert len(ids)==len(set(ids))==720 and not set(ids)&blocked
  previous=json.loads((PRIOR/('U%d'%u)/'results_public.json').read_text())
  attacks=previous['reference']['baseline_rbf']['attack_by_user'];hardness=[]
  for v,b in attacks.items():
   hardness.append(dict(user=int(v),prior_test_far=float(b['accepted']/b['n']),prior_test_accepted=b['accepted'],prior_test_n=b['n']))
  hardness.sort(key=lambda a:(-a['prior_test_far'],a['user']))
  save(folder/'ranking_test_informed.json',dict(target=u,ranking=hardness,scope='Explicitly selected using prior final attack outcomes; descriptive post-hoc subset only',test_informed=True))
  models={};locked={};private_oof={}
  for drop in [0,1,2]:
   removed=[a['user'] for a in hardness[:drop]];mask=~np.isin(labels,np.asarray(removed)-1)
   labs=labels[mask];yy=labs==u-1;ww=words[mask]
   assert yy.sum()==80 and (~yy).sum()==(8-drop)*80
   folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(yy)),yy,ww))
   calmask=~np.isin(p.labels['impostor_calibration'],np.asarray(removed)-1)
   for feature in ['base','identity_acc']:
    x=np.concatenate([dev[k+'__'+feature] for k in ('auth_train','model_validation')])[mask]
    c,gamma=old.sel.choose_params(x,yy,ww,71);rbf=old.sel.rbf.fit([x],yy,c,gamma)
    sg=old.sel.rbf.score(rbf,[dev['target_dev_pool__'+feature]]);sn=old.sel.rbf.score(rbf,[dev['impostor_calibration__'+feature][calmask]])
    key='%s_drop%d_rbf'%(feature,drop);locked[key]=dict(feature=feature,removed=removed,drop=drop,method='rbf',C=c,gamma=gamma,threshold=float(old.base.original.choose_threshold(sg,sn)));models[key]=rbf
    aa=np.zeros(len(yy));bb=np.zeros(len(yy))
    for tr,va in folds:
     assert not set(ww[tr])&set(ww[va]);aa[va],bb[va]=proto_pair(proto_fit(x[tr],labs[tr]),x[va],u)
    candidates=[]
    for lam in LAMBDAS:
     values=[float(roc_auc_score(yy[va],(aa-lam*bb)[va])) for _,va in folds];candidates.append(dict(lam=lam,mean_auc=float(np.mean(values)),fold_aucs=values))
    lam=min(candidates,key=lambda a:(-a['mean_auc'],a['lam']))['lam'];model=proto_fit(x,labs)
    a,b=proto_pair(model,dev['target_dev_pool__'+feature],u);g=a-lam*b
    a,b=proto_pair(model,dev['impostor_calibration__'+feature][calmask],u);n=a-lam*b
    key='%s_drop%d_relative'%(feature,drop);locked[key]=dict(feature=feature,removed=removed,drop=drop,method='relative',lam=lam,candidates=candidates,threshold=float(old.base.original.choose_threshold(g,n)));models[key]=model
    private_oof[key+'__target']=aa;private_oof[key+'__background']=bb;private_oof[key+'__labels']=labs;private_oof[key+'__words']=ww
  save(folder/'locked_before_queries.json',dict(target=u,routes=locked,selection_test_informed=True,new_scores_read=False))
  old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**private_oof)
  final=dict(np.load(str(PRIOR/('U%d'%u)/'final_features_private.npz')));rows={};dump={}
  for key,spec in locked.items():
   ss={}
   for role in ('final_genuine','final_impostor'):
    x=final[role+'__'+spec['feature']]
    if spec['method']=='rbf':s=old.sel.rbf.score(models[key],[x])
    else:a,b=proto_pair(models[key],x,u);s=a-spec['lam']*b
    ss[role]=s;dump[key+'__'+role]=s
   result=old.sel.evaluate_route(ss,p,u,spec['threshold']);result.update({k:v for k,v in spec.items() if k!='candidates'})
   for name,mask in [('removed_attackers',np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1)),('retained_attackers',~np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1))]:
    score=ss['final_impostor'][mask];result[name]=dict(n=len(score),accepted=int((score>=spec['threshold']).sum()),far=float((score>=spec['threshold']).mean()) if len(score) else None)
   all_diagnostic=result
   keep=~np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1)
   import types
   limited_protocol=types.SimpleNamespace(labels={'final_impostor':p.labels['final_impostor'][keep]})
   limited=old.sel.evaluate_route({'final_genuine':ss['final_genuine'],'final_impostor':ss['final_impostor'][keep]},limited_protocol,u,spec['threshold'])
   limited.update({k:v for k,v in spec.items() if k!='candidates'})
   limited['test_informed_subset']=True
   limited['all_attackers_diagnostic']=all_diagnostic
   rows[key]=limited
  np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
  out=dict(target=u,ranking=hardness,routes=rows);save(folder/'results_public.json',out);allrows[str(u)]=out;save(ROOT/'results_public.json',allrows)
  print('TARGET_DONE',u,'removed', [a['user'] for a in hardness[:2]],flush=True)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(allrows)))
if __name__=='__main__':main()

