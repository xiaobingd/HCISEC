"""Formal independent-top-K AS-Norm1 on fixed train-only speaker prototypes."""
import sys,json,os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_posthoc_subset_20261029 as previous
old=previous.old;PRIOR=previous.ROOT
ROOT=Path('/autodl-fs/data/mei_formal_asnorm_20261029_f64')
STD_FLOOR=1e-4
def asnorm(e,t,cohort,k):
 e=np.asarray(e,dtype=np.float64);t=np.asarray(t,dtype=np.float64);cohort=np.asarray(cohort,dtype=np.float64)
 assert 2<=k<=len(cohort)
 raw=t@e;ez=np.sort(cohort@e)[-k:];tz=np.sort(t@cohort.T,axis=1)[:,-k:]
 return .5*((raw-ez.mean())/max(ez.std(ddof=1),STD_FLOOR)+(raw-tz.mean(1))/np.maximum(tz.std(1,ddof=1),STD_FLOOR))
def vectors(model,x,u):
 scaler,uids,centers=model;return centers[uids==u-1].ravel(),previous.norm(scaler.transform(x)),centers[uids!=u-1]
def save(p,x):previous.save(p,x)
def selfcheck():
 rng=np.random.RandomState(42);x=previous.norm(rng.normal(size=(12,7)));e=x[0];t=x[1];c=x[2:]
 assert np.allclose(asnorm(e,t[None],c,4),asnorm(t,e[None],c,4),rtol=0,atol=1e-12)
 scores=c@e;test=c@t;raw=e@t
 expected=.5*((raw-np.mean(sorted(scores)[-4:]))/np.std(sorted(scores)[-4:],ddof=1)+(raw-np.mean(sorted(test)[-4:]))/np.std(sorted(test)[-4:],ddof=1))
 assert abs(expected-asnorm(e,t[None],c,4)[0])<1e-12
def main():
 selfcheck();ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(method='Formal AS-Norm1: half of enrollment-topK Z-normalization plus test-topK T-normalization, independently selected cohorts',formula='0.5*((s-mu_enroll_topK)/sigma_enroll_topK+(s-mu_test_topK)/sigma_test_topK)',score_precision='float64',std_ddof=1,std_floor=STD_FLOOR,fixed_K=4,K_grid='integers2 through min(5,background_count-1)',K_selection='5 train content-fold mean AUC, ties larger K;final not used',cohort='one training-only normalized mean prototype per retained background identity;target excluded',fit_per_user=80,drop_counts=[0,1,2],features=['base772','identity_acc64'],upstream_unchanged=True,limitations=['6 to8 background identities only','Drop1/2 lists were previously chosen using FINAL results; restricted-cohort estimates are post-hoc not independent','All identities known upstream; not open-set','Single seed, historical final data reused'],references=['https://www.fit.vut.cz/research/group/speech/public/publi/2017/matejka_interspeech2017_IS170803.pdf','https://github.com/Snowdar/asv-subtools/blob/master/score/ScoreNormalization.py'],source_sha256=old.sha(Path(__file__))))
 rows={}
 for u in old.ORDER:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)))
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);p=old.protocol(u)
  source=PRIOR/('U%d'%u);ref=json.loads((source/'results_public.json').read_text());fitted=old.sel.joblib.load(str(source/'models_private.joblib'))
  dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'))
  labels=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')])
  words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
  locks={};models={};oo={}
  for drop in [0,1,2]:
   removed=[a['user'] for a in ref['ranking'][:drop]];mask=~np.isin(labels,np.asarray(removed)-1);lab=labels[mask];y=lab==u-1;ww=words[mask]
   assert len(lab)==80*(9-drop) and y.sum()==80
   folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,ww));ks=list(range(2,min(5,7-drop)+1))
   calmask=~np.isin(p.labels['impostor_calibration'],np.asarray(removed)-1)
   for feature in ['base','identity_acc']:
    x=np.concatenate([dev[role+'__'+feature] for role in ('auth_train','model_validation')])[mask];model=fitted['%s_drop%d_relative'%(feature,drop)]
    oof={k:np.zeros(len(y)) for k in ks}
    for tr,va in folds:
     assert not set(ww[tr])&set(ww[va]);m=previous.proto_fit(x[tr],lab[tr]);e,t,c=vectors(m,x[va],u)
     for k in ks:oof[k][va]=asnorm(e,t,c,k)
    cv=[dict(k=k,mean_auc=float(np.mean([roc_auc_score(y[va],score[va]) for _,va in folds]))) for k,score in oof.items()]
    best=min(cv,key=lambda a:(-a['mean_auc'],-a['k']))['k']
    for k,score in oof.items():oo['%s_drop%d_K%d'%(feature,drop,k)]=score
    for variant,k in [('asnorm_fixed4',4),('asnorm_selected',best),('snorm',8-drop)]:
     key='%s_drop%d_%s'%(feature,drop,variant);cal={}
     for role in ['target_dev_pool','impostor_calibration']:
      a=dev[role+'__'+feature];a=a if role=='target_dev_pool' else a[calmask]
      e,t,c=vectors(model,a,u);cal[role]=asnorm(e,t,c,k)
     threshold=float(old.base.original.choose_threshold(cal['target_dev_pool'],cal['impostor_calibration']))
     locks[key]=dict(feature=feature,drop=drop,removed=removed,k=k,variant=variant,threshold=threshold,cv=cv);models[key]=model
  save(folder/'locked_before_queries.json',dict(target=u,routes=locks,new_scores_read=False,drop_lists_test_informed=True));np.savez_compressed(str(folder/'oof_private.npz'),**oo)
  final=np.load(str(previous.PRIOR/('U%d'%u)/'final_features_private.npz'));results={};dump={}
  for key,spec in locks.items():
   ss={}
   for role in ['final_genuine','final_impostor']:
    e,t,c=vectors(models[key],final[role+'__'+spec['feature']],u);ss[role]=asnorm(e,t,c,spec['k']);dump[key+'__'+role]=ss[role]
   full=old.sel.evaluate_route(ss,p,u,spec['threshold']);keep=~np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1)
   import types
   restricted=old.sel.evaluate_route({'final_genuine':ss['final_genuine'],'final_impostor':ss['final_impostor'][keep]},types.SimpleNamespace(labels={'final_impostor':p.labels['final_impostor'][keep]}),u,spec['threshold'])
   restricted.update({k:v for k,v in spec.items() if k!='cv'});restricted['all_attackers_diagnostic']=full
   results[key]=restricted
  np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor']);save(folder/'results_public.json',dict(target=u,routes=results,reference=ref))
  rows[str(u)]=dict(target=u,routes=results,reference=ref);save(ROOT/'results_public.json',rows);print('TARGET_DONE',u,flush=True)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':main()

